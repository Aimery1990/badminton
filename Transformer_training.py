import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
# from torchsummary import summary
from torchinfo import summary


from Assignment4_DataProcess import Keywords, run_data_processing_transformer


class SequenceDataset(Dataset):
    def __init__(self, windows, labels, label_encoder=None):
        self.X = torch.tensor(windows, dtype=torch.float32)   # (N, 20, 9)
        self.labels_raw = labels

        # Encode labels to 0~3
        if label_encoder is None:
            self.le = LabelEncoder()
            self.y = torch.tensor(self.le.fit_transform(labels), dtype=torch.long)
        else:
            self.le = label_encoder
            self.y = torch.tensor(self.le.transform(labels), dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class PositionalEncodingLearned(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, seq_len, d_model))

    def forward(self, x):
        return x + self.pe


class TransformerShotClassifier(nn.Module):
    def __init__(self, input_dim=9, seq_len=20, d_model=64, nhead=4, num_layers=2, num_classes=4):
        super().__init__()

        # Input embedding (9 -> 64)
        self.fc_in = nn.Linear(input_dim, d_model)

        # Positional encoding (learnable)
        self.pos_enc = PositionalEncodingLearned(seq_len, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1,
            activation="relu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):  # x: (batch, 20, 9)
        x = self.fc_in(x)        # -> (batch, 20, 64)
        x = self.pos_enc(x)      # add learnable positional encoding
        x = self.transformer(x)  # -> (batch, 20, 64)

        # Global average pooling over time axis
        x = x.mean(dim=1)        # -> (batch, 64)

        return self.classifier(x)



class TransformerTrainer:
    def __init__(self):

        train_win, train_lab, test_win, test_lab = run_data_processing_transformer()

        self.train_windows = train_win
        self.test_windows = test_win
        self.train_windows_labels = train_lab
        self.test_windows_labels = test_lab

        self.train_dataset = SequenceDataset(self.train_windows, self.train_windows_labels, label_encoder=None)
        self.test_dataset = SequenceDataset(self.test_windows, self.test_windows_labels, self.train_dataset.le)

        self.train_loader = DataLoader(self.train_dataset, batch_size=32, shuffle=True)
        self.val_loader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerShotClassifier().to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5
        )

        # self.EPOCHS = 40
        # self.early_stop_patience = 6
        # self.best_val_loss = float('inf')
        self.early_stop_counter = 0

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []


    def train(self, epochs=40, early_stop_patience=6,best_val_loss=float('inf')):
        for epoch in range(epochs):
            self.model.train()
            train_preds, train_targets = [], []
            train_loss_sum = 0

            for X, y in self.train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(X)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                train_loss_sum += loss.item()
                train_preds.extend(logits.argmax(1).cpu().numpy())
                train_targets.extend(y.cpu().numpy())

            train_loss = train_loss_sum / len(self.train_loader)
            train_acc = accuracy_score(train_targets, train_preds)

            # ===== Validation =====
            self.model.eval()
            val_preds, val_targets = [], []
            val_loss_sum = 0

            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.model(X)
                    loss = self.criterion(logits, y)
                    val_loss_sum += loss.item()

                    val_preds.extend(logits.argmax(1).cpu().numpy())
                    val_targets.extend(y.cpu().numpy())

            val_loss = val_loss_sum / len(self.val_loader)
            val_acc = accuracy_score(val_targets, val_preds)

            # record
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"Epoch {epoch + 1}/{epochs} | "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
                  f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

            # scheduler
            self.scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), "best_transformer_model.pth")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= early_stop_patience:
                    print("Early stopping triggered!")
                    break


    def plot_training_loss(self):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")

    def plot_validation_loss(self):
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Validation Acc')
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.show()


    def best_model_verify(self):
        # Reload best model
        best_model = TransformerShotClassifier()
        best_model.load_state_dict(torch.load("best_transformer_model.pth"))
        best_model.to(self.device)
        best_model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in self.val_loader:
                logits = best_model(X.to(self.device))
                preds = logits.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(y.numpy())


        print(classification_report(all_targets, all_preds, target_names=self.train_dataset.le.classes_))

        cm = confusion_matrix(all_targets, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.train_dataset.le.classes_)
        disp.plot(cmap='Blues')
        plt.show()

        summary(self.model, input_size=(1, 20, 9), depth=3)

if __name__ == "__main__":

    print("\n========== Running Transformer Sequence Model Training ==========\n")

    # Create trainer (this automatically loads + processes your dataset)
    trainer = TransformerTrainer()

    # Train Model B (Transformer)
    trainer.train(
        epochs=40,
        early_stop_patience=6,
        best_val_loss=float('inf')
    )

    # Plot Training Curves
    trainer.plot_training_loss()
    trainer.plot_validation_loss()

    # Evaluate best model
    print("\n========== Evaluating Best Transformer Model ==========\n")
    trainer.best_model_verify()

    print("\n========== Training Complete ==========\n")
