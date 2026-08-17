import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from PreprocessParams import TARGET_FRAMES, FREQUENCY_BIN_COUNT, downsampled_size
from Visualizations import plot_loss_per_epoch, plot_accuracy_per_epoch, plot_confusion_matrix
from audio_dataset import EmotionSpecDataset

"""
for extracting meta-data for organized plot savings
"""
import inspect
from collections import OrderedDict
from typing import Any

from results_manager import ResultsManager

# The book reports a single 40-epoch run per dataset. Without a fixed seed the
# weight init, the DataLoader shuffling and the train/val/test split all move,
# and none of the reported numbers can be reproduced.
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed every RNG the training pipeline touches."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class SentimentModelHandler:
    """
    Wrapper class for general model hyperparameters.
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        test_dataset,
        raw_data_class_name: str,
        **kwargs,
    ):

        # ------------------------------------------------------------------
        # 1. Regular initialisation
        # ------------------------------------------------------------------

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model: nn.Module = model

        self._train_dataset: EmotionSpecDataset = train_dataset
        self._val_dataset: EmotionSpecDataset = val_dataset
        self._test_dataset: EmotionSpecDataset = test_dataset

        self._train_class_names = self._train_dataset.class_names
        self._val_class_names = self._val_dataset.class_names
        self._test_class_names = self._test_dataset.class_names

        self._batch_size: int = kwargs.get('batch_size', 16)
        self._lr: float = kwargs.get("learning_rate", 0.1)

        self._train_loader: DataLoader = DataLoader(self._train_dataset, self._batch_size, shuffle=True)
        self._val_loader: DataLoader = DataLoader(self._val_dataset, self._batch_size, shuffle=False)
        self._test_loader: DataLoader = DataLoader(self._test_dataset, self._batch_size, shuffle=False)

        self._class_weights = train_dataset.class_weights
        print(f"Training set class Weights: {self._class_weights}")

        self._criterion = kwargs.get("criterion", nn.CrossEntropyLoss)(weight=self._class_weights.to(self._device))
        # Book, section 4.2.1: SGD, momentum 0.9, L2 weight decay 1e-6.
        self._optimizer = kwargs.get("optimizer", optim.SGD)(self._model.parameters(), lr=self._lr, momentum=0.9, weight_decay=1e-6)
        # Book, section 4.2.1: lr = 1e-3 until the 33rd epoch, then 1e-4. The
        # second milestone only matters for runs longer than the reported 40
        # epochs; it is kept so the schedule is explicit rather than implied by
        # `int(0.33 * 100)`.
        self._lr_milestones: list[int] = kwargs.get("lr_milestones", [33, 66])
        self._scheduler = kwargs.get("scheduler", optim.lr_scheduler.MultiStepLR)(self._optimizer, milestones=self._lr_milestones, gamma=0.1)

        self._training_logs: dict = dict()

        # Tuple structure is (Loss score, Accuracy score)
        self._train_scores: list[tuple[float, float]] = []
        self._val_scores: list[tuple[float, float]] = []
        self._test_score: float = 0.0

        # New lists to store the true labels and predicted labels for confusion matrix(Of both train and validation).
        self._true_labels_train, self._true_labels_val, self._true_labels_test = [], [], []
        self._pred_labels_train, self._pred_labels_val, self._pred_labels_test = [], [], []

        """
        for extracting meta-data for organized plot savings
        """
        # 1) Keep the hyper-parameters exactly as passed
        self.hparams: OrderedDict[str, Any] = OrderedDict(kwargs)
         # 2) Optional – auto-collect the *model* constructor kwargs
        sig = inspect.signature(self._model.__class__.__init__)
        ctor_args = {
            k: v.default
            for k, v in sig.parameters.items()
            if k != "self" and v.default is not inspect._empty
        }
        self.hparams.update({f"model_{k}": v for k, v in ctor_args.items()})
        

        # ------------------------------------------------------------------
        # 2. Create run-specific results directory
        # ------------------------------------------------------------------
        self._results = ResultsManager(
            model_class=model.__class__.__name__,
            raw_data_class=raw_data_class_name,
            dataset_class=train_dataset.__class__.__name__,
            user_notes=kwargs.get("run_notes"),
        )


    def __train_one_epoch(self):
        """
        Trains a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
        temp_true_labels_train = []
        temp_pred_labels_train = []

        self._model.train()
        running_loss = 0.0
        correct = 0

        for mel_spec, label in tqdm(self._train_loader, desc="Model training"):
            mel_spec, label = mel_spec.to(self._device), label.to(self._device)

            self._optimizer.zero_grad()
            output = self._model(mel_spec)
            #print("Output sample:", output[0].detach().cpu().numpy())
            #print("Label sample:", label[0].item())
            loss = self._criterion(output, label)
            loss.backward()
            self._optimizer.step()

            running_loss += loss.item() * mel_spec.size(0)

            predictions = output.argmax(1)
            correct += (predictions == label).sum().item()

            temp_true_labels_train.extend(label.cpu().numpy())
            temp_pred_labels_train.extend(predictions.cpu().numpy())

        # update the main lists after epoch ends
        self._true_labels_train = temp_true_labels_train
        self._pred_labels_train = temp_pred_labels_train
        
        self._scheduler.step()
        total_samples = len(self._train_loader.dataset)

        return running_loss / total_samples, correct, total_samples

    def __validate(self):
        """
        Validates a single epoch across a dataloader.
        @return: Loss average across batches, number of correct classifications and total samples.
        """
        temp_true_labels_val = []
        temp_pred_labels_val = []

        self._model.eval()
        running_loss = 0.0
        correct = 0


        with torch.no_grad():
            for mel_spec, label in tqdm(self._val_loader, desc="Model validation"):
                mel_spec, label = mel_spec.to(self._device), label.to(self._device)
                output = self._model(mel_spec)
                loss = self._criterion(output, label)

                running_loss += loss.item() * mel_spec.size(0)

                predictions = output.argmax(1)
                correct += (predictions == label).sum().item()

                temp_true_labels_val.extend(label.cpu().numpy())
                temp_pred_labels_val.extend(predictions.cpu().numpy())
        
        # update the main lists after epoch ends
        self._true_labels_val = temp_true_labels_val
        self._pred_labels_val = temp_pred_labels_val
        
        total_samples = len(self._val_loader.dataset)
        return running_loss / total_samples, correct, total_samples

    def __test(self) -> float:
        self._model.eval()
        running_loss = 0.0
        correct = 0

        # Reset first: __test may be called more than once (plot_accuracies and
        # report_classification both need it) and appending twice would double
        # every count in the test confusion matrix.
        self._true_labels_test, self._pred_labels_test = [], []

        with torch.no_grad():
            for mel_spec, label in tqdm(self._test_loader, desc="Model test"):
                mel_spec, label = mel_spec.to(self._device), label.to(self._device)
                output = self._model(mel_spec)
                loss = self._criterion(output, label)

                running_loss += loss.item() * mel_spec.size(0)

                predictions = output.argmax(1)
                correct += (predictions == label).sum().item()

                self._true_labels_test.extend(label.cpu().numpy())
                self._pred_labels_test.extend(predictions.cpu().numpy())

        total_samples = len(self._test_loader.dataset)
        return running_loss / total_samples, correct, total_samples

    def train_model(self, epochs: int = 10, verbose: bool = False):
        """
        Applies the entire training logic and saves the results.
        @param epochs: Number of epochs to train the model. Defaults to 10.
        @param verbose: Verbosity of results. Defaults to False.
        """
        self._model.to(self._device)

        print(f"Training model with device: {self._device}")
        for epoch in range(epochs):
            train_loss, train_correct, train_total = self.__train_one_epoch() # noam question eli: does train loss is the average train loss per batch?
            val_loss, val_correct, val_total = self.__validate()

            train_accuracy = (train_correct / train_total) * 100.0
            val_accuracy = (val_correct / val_total) * 100.0
            current_lr = self._optimizer.param_groups[0]['lr']

            epoch_string: str = f"Epoch {epoch + 1}"
            results_string: str = f'Train Loss: {train_loss:.4f},\n' \
                                  f'Train Accuracy: {train_accuracy:.2f}%\n' \
                                  f'[{train_correct}/{train_total}]\n' \
                                  f'Val Loss: {val_loss:.4f},\n' \
                                  f'Val Accuracy: {val_accuracy:.2f}%\n' \
                                  f'[{val_correct}/{val_total}]\n' \
                                  f'LR: {current_lr}'

            self._training_logs[epoch_string] = results_string
            self._train_scores.append((train_loss, train_accuracy))
            self._val_scores.append((val_loss, val_accuracy))

            if verbose:
                print(f"Epoch {epoch + 1}")
                print(results_string)
                print("--------------------------------\n")

        

    def __str__(self):
        return (f"Model name: {self._model.__class__.__name__}\n"
                f"Device: {self._device}\n"
                f"Batch size: {self._batch_size}\n"
                f"Criterion: {self._criterion.__class__.__name__}\n"
                f"Optimizer: {self._optimizer.__class__.__name__}")

    """Properties"""

    @property
    def model_name(self) -> str:
        return self._model.__class__.__name__.__str__()

    @property
    def batch_size(self) -> str:
        return self._batch_size.__str__()

    @property
    def starting_lr(self) -> str:
        return self._lr.__str__()

    @property
    def criterion(self) -> str:
        return self._criterion.__class__.__name__.__str__()

    @property
    def optimizer(self) -> str:
        return self._optimizer.__class__.__name__.__str__()

    def __generate_plot_title(self) -> str:
        return (f"{self.model_name}_"
                f"")

    def ask_to_save_model(self, option: str = 'default'):
        if option != 'default':
            user_input = input("what name to save the model? ('n' to avoid saving, 'd' to use default): ")
        else: 
            user_input = 'd'
        if user_input.lower() == 'n':
            print("Model not saved.")
        elif user_input.lower() == 'd':
            torch.save(self._model, Path(os.path.join(self._results.base_dir, f"{self._model.__class__.__name__}.pt")), _use_new_zipfile_serialization=True)
        else:
            torch.save(self._model, Path(os.path.join(self._results.base_dir, f"{user_input}.pt")), _use_new_zipfile_serialization=True)

    def plot_losses(self, file_name: str | None = None):
        file_name = file_name or f"{self._model.__class__.__name__}_losses"
        train_losses = [scores[0] for scores in self._train_scores]
        val_losses = [scores[0] for scores in self._val_scores]
        self._results.create_directory()
        plot_loss_per_epoch(file_save_name= file_name,
                            hparams=self.hparams,              # ← lives in the handler
                            training_loss=train_losses,
                            validation_loss=val_losses,
                            dir_path=self._results.base_dir)

    def plot_accuracies(self, file_name: str | None = None):
        file_name = file_name or f"{self._model.__class__.__name__}_accuracies"
        train_accuracies = [scores[1] for scores in self._train_scores]
        val_accuracies = [scores[1] for scores in self._val_scores]
        self._results.create_directory()
        _, test_correct, test_total = self.__test()
        self._test_score = test_correct / test_total
        print("Test score", self._test_score)
        plot_accuracy_per_epoch(file_save_name=file_name,
                                hparams=self.hparams,              # ← lives in the handler
                            training_accuracy=train_accuracies,
                            validation_accuracy=val_accuracies,
                            test_accuracy=self._test_score,
                            dir_path=self._results.base_dir)

    def plot_confusion_matrix(self, file_name: str | None = None):
        file_name = file_name or f"{self._model.__class__.__name__}_confusion_matrix"
        if not self._true_labels_test:
            # The test split is only evaluated inside __test(); make the call
            # order-independent so the Test panel is never silently dropped.
            self.__test()
        train_confusion_data = (self._true_labels_train, self._pred_labels_train, self._train_class_names)
        val_confusion_data = (self._true_labels_val, self._pred_labels_val, self._val_class_names)
        test_confusion_data = (self._true_labels_test, self._pred_labels_test, self._test_class_names)
        self._results.create_directory()
        plot_confusion_matrix(file_save_name=file_name,
                              hparams=self.hparams,              # ← lives in the handler
                              train_label_data=train_confusion_data,
                              val_label_data=val_confusion_data,
                              test_label_data=test_confusion_data,
                              dir_path=self._results.base_dir)

    def report_classification(self, file_name: str = "classification_report") -> dict[str, dict]:
        """Per-class and averaged precision / recall / F1 for every split.

        This is the source of the tables in the project book: Table 2 (weighted
        averages, depth vs single-input), Table 3 (per class, RAVDESS),
        Tables 8-9 (TESS) and Tables 10-11 (CREMA-D). It was previously produced
        by hand, so nothing in the repository regenerated it.

        Writes `<file_name>.csv` (tidy, one row per class/average per split) and
        `<file_name>.txt` (the human-readable sklearn report) into the run
        directory, and returns the raw report dictionaries.
        """
        import pandas as pd

        if not self._true_labels_test:
            self.__test()

        self._results.create_directory()

        splits = {
            "train": (self._true_labels_train, self._pred_labels_train, self._train_class_names),
            "validation": (self._true_labels_val, self._pred_labels_val, self._val_class_names),
            "test": (self._true_labels_test, self._pred_labels_test, self._test_class_names),
        }

        reports: dict[str, dict] = {}
        rows: list[dict] = []
        text_blocks: list[str] = []

        for split_name, (y_true, y_pred, class_names) in splits.items():
            if not y_true:
                continue
            labels = list(range(len(class_names)))
            report = classification_report(
                y_true, y_pred,
                labels=labels,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            reports[split_name] = report
            text_blocks.append(
                f"===== {split_name} =====\n"
                + classification_report(y_true, y_pred, labels=labels,
                                        target_names=class_names, zero_division=0)
            )
            for key, value in report.items():
                if not isinstance(value, dict):   # 'accuracy' is a bare float
                    continue
                rows.append({
                    "split": split_name,
                    "class": key,
                    "precision": round(value["precision"], 3),
                    "recall": round(value["recall"], 3),
                    "f1-score": round(value["f1-score"], 3),
                    "support": int(value["support"]),
                })
            rows.append({
                "split": split_name, "class": "accuracy",
                "precision": "", "recall": "",
                "f1-score": round(report["accuracy"], 3),
                "support": int(report["macro avg"]["support"]),
            })

        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(self._results.base_dir, f"{file_name}.csv"), index=False)
        with open(os.path.join(self._results.base_dir, f"{file_name}.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n\n".join(text_blocks))

        return reports

    def save_plots(self):
        #TODO: Add robust title for plots for easy identification
        self.plot_accuracies()
        self.plot_losses()
        self.plot_confusion_matrix()
        self.report_classification()


"""Emo-Net Logic"""

class ResidualBlockNew(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=False):
        super(ResidualBlockNew, self).__init__()

        self.stride = stride
        self.shortcut = shortcut

        # First convolution
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Second convolution
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if self.shortcut:
            self.shortcut_pool = nn.AvgPool2d(2, stride=2, ceil_mode=True)
            self.shortcut_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)


    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        #print(f"{self.shortcut}")
        if self.shortcut:
            identity = self.shortcut_pool(identity)
            identity = self.shortcut_conv(identity)

        out += identity
        out = self.relu(out)
        return out


class ResNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks, stride=1):
        super(ResNetModule, self).__init__()

        self.blocks = []
        for i in range(num_blocks):
            if i == 0:  # First block requires a shortcut connection
                self.blocks.append(ResidualBlockNew(in_channels, out_channels, stride=stride, shortcut=True))
            else:
                self.blocks.append(ResidualBlockNew(out_channels, out_channels, stride=1, shortcut=False))

        self.blocks = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.blocks(x)


class ResNetWithAttention(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttention, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # First submodule with 64 filters
        self.module1 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(64, 128, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(128, 256, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        # The three ResNet modules each halve both spatial dimensions with a
        # stride-2 3x3 convolution, i.e. n -> ceil(n / 2). That is only the same
        # as `n // 8` when n is divisible by 8, so the size is derived with
        # downsampled_size() instead - otherwise a duration change silently
        # produces a shape mismatch at the first FC layer.
        flattened_features = 256 * downsampled_size(TARGET_FRAMES) * downsampled_size(FREQUENCY_BIN_COUNT)
        self.fc1 = nn.Linear(flattened_features, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))

        # Output layer (classification)
        x = self.fc_out(x)

        return x

# """ResNet 2 channels"""
class ResNetWithAttention2d(nn.Module):
    def __init__(self, num_classes=8):
        super(ResNetWithAttention2d, self).__init__()

        # Initial convolutional block
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # First submodule with 64 filters
        self.module1 = ResNetModule(32, 64, num_blocks=2, stride=2)

        # Second submodule with 128 filters
        self.module2 = ResNetModule(64, 128, num_blocks=2, stride=2)

        # Third submodule with 256 filters
        self.module3 = ResNetModule(128, 256, num_blocks=2, stride=2)

        # Attention layer (Self-Attention)
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)

        # Final batch normalization and ReLU
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # Fully connected layers (FC layers)
        # The three ResNet modules each halve both spatial dimensions with a
        # stride-2 3x3 convolution, i.e. n -> ceil(n / 2). That is only the same
        # as `n // 8` when n is divisible by 8, so the size is derived with
        # downsampled_size() instead - otherwise a duration change silently
        # produces a shape mismatch at the first FC layer.
        flattened_features = 256 * downsampled_size(TARGET_FRAMES) * downsampled_size(FREQUENCY_BIN_COUNT)
        self.fc1 = nn.Linear(flattened_features, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)

        # Output layer (final classification layer)
        self.fc_out = nn.Linear(512, num_classes)

    def forward(self, x):
        # Initial convolution
        x = self.relu(self.bn1(self.conv1(x)))

        # Pass through the residual modules
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)

        # Apply attention
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, channels, -1).transpose(1, 2)  # Flatten the spatial dimensions
        x, _ = self.attention(x, x, x)
        x = x.transpose(1, 2).view(batch_size, channels, height, width)  # Reshape back to 4D

        # Final batch normalization and ReLU activation
        x = self.relu(self.bn2(x))

        # Flatten for FC layers
        x = x.reshape(x.size(0), -1)  # Flatten the tensor

        # First FC layer
        x = self.relu(self.bn_fc1(self.fc1(x)))

        # Second FC layer
        x = self.relu(self.bn_fc2(self.fc2(x)))

        # Output layer (classification)
        x = self.fc_out(x)

        return x