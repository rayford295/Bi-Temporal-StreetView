# ==============================================================================
# Dual-Swin Bi-Temporal Classifier Training with Progressive Split Ratios
# ==============================================================================

import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict
import os
from PIL import Image
from tqdm import tqdm
import time
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. Reproducibility Setup
# ==============================================================================
def set_seed(seed=42):
    """Set random seeds for improved reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# ==============================================================================
# 2. Device Setup
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==============================================================================
# 3. Image Preprocessing
# ==============================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ==============================================================================
# 4. Dataset Definition (Pre/Post Image Pairs)
# ==============================================================================
class DisasterDataset(Dataset):
    """
    A bi-temporal dataset loader. Each sample contains:
      - pre-disaster image (e.g., year 2023 in filename)
      - post-disaster image (e.g., year 2024 in filename)
      - class label inferred from folder naming convention (e.g., *_0, *_1, *_2)
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        print("Loading dataset...")
        for class_folder in tqdm(sorted(os.listdir(root_dir)), desc="Processing classes"):
            class_path = os.path.join(root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue

            for folder in os.listdir(class_path):
                image_folder = os.path.join(class_path, folder)
                if not os.path.isdir(image_folder):
                    continue

                images = sorted(os.listdir(image_folder))
                if len(images) != 2:
                    continue

                pre_image, post_image = None, None
                for img in images:
                    if "2023" in img:
                        pre_image = os.path.join(image_folder, img)
                    elif "2024" in img:
                        post_image = os.path.join(image_folder, img)

                if pre_image and post_image:
                    label = int(class_folder.split("_")[-1])
                    self.data.append((pre_image, post_image, label))

        print(f"Dataset loaded successfully. Total samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pre_image_path, post_image_path, label = self.data[idx]

        try:
            pre_image = Image.open(pre_image_path).convert("RGB")
            post_image = Image.open(post_image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading images at index {idx}: {e}")
            raise e

        if self.transform:
            pre_image = self.transform(pre_image)
            post_image = self.transform(post_image)

        return pre_image, post_image, label

# ==============================================================================
# 5. Model Definition (Dual Swin + Feature Fusion)
# ==============================================================================
class DualSwinTransformerClassifier(nn.Module):
    """
    Dual-stream Swin Transformer:
      - One Swin backbone for pre-disaster images
      - One Swin backbone for post-disaster images
    Features are concatenated and fused before classification.
    """

    def __init__(self, num_classes=3):
        super(DualSwinTransformerClassifier, self).__init__()
        print("Loading Swin Transformer backbones...")

        self.swin_pre = timm.create_model("swin_base_patch4_window7_224",
                                          pretrained=True, num_classes=0)
        self.swin_post = timm.create_model("swin_base_patch4_window7_224",
                                           pretrained=True, num_classes=0)

        # Freeze backbone weights (train only fusion + classifier by default)
        for param in self.swin_pre.parameters():
            param.requires_grad = False
        for param in self.swin_post.parameters():
            param.requires_grad = False

        feature_dim = self.swin_pre.num_features

        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier = nn.Linear(feature_dim, num_classes)

        print("Model initialization completed.")

    def forward(self, pre_image, post_image):
        pre_features = self.swin_pre(pre_image)
        post_features = self.swin_post(post_image)
        combined = torch.cat((pre_features, post_features), dim=1)
        fused = self.fusion(combined)
        output = self.classifier(fused)
        return output

# ==============================================================================
# 6. Training and Evaluation
# ==============================================================================
def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                       epochs=20, patience=3):
    """
    Train with early stopping based on validation accuracy.
    Saves the best model weights to 'best_model.pth'.
    """
    best_accuracy = 0.0
    patience_counter = 0

    train_losses = []
    val_accuracies = []
    class_accuracies_list = []

    for epoch in range(epochs):
        # -------------------------
        # Training
        # -------------------------
        model.train()
        running_loss = 0.0
        correct_train, total_train = 0, 0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for pre_images, post_images, labels in train_pbar:
            pre_images = pre_images.to(device)
            post_images = post_images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pre_images, post_images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            train_pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100.0 * correct_train / total_train:.2f}%"
            })

        train_loss = running_loss / max(len(train_loader), 1)
        train_accuracy = 100.0 * correct_train / max(total_train, 1)
        train_losses.append(train_loss)

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        correct_val, total_val = 0, 0
        all_preds, all_labels = [], []

        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")
        with torch.no_grad():
            for pre_images, post_images, labels in val_pbar:
                pre_images = pre_images.to(device)
                post_images = post_images.to(device)
                labels = labels.to(device)

                outputs = model(pre_images, post_images)
                _, predicted = torch.max(outputs, 1)

                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                for i in range(len(labels)):
                    y = labels[i].item()
                    p = predicted[i].item()
                    if y == p:
                        class_correct[y] += 1
                    class_total[y] += 1

        val_accuracy = 100.0 * correct_val / max(total_val, 1)
        val_accuracies.append(val_accuracy)

        class_accuracies = {
            f"Class {c} Accuracy": (100.0 * class_correct[c] / class_total[c]) if class_total[c] > 0 else 0.0
            for c in class_total
        }
        class_accuracies_list.append(class_accuracies)

        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0) * 100.0
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0) * 100.0
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100.0

        print(
            f"Epoch {epoch+1}: "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
            f"Val Acc: {val_accuracy:.2f}%, P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}"
        )
        print(f"Class-wise Validation Accuracy: {class_accuracies}")

        # -------------------------
        # Early Stopping
        # -------------------------
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        # ReduceLROnPlateau expects a monitored metric (validation accuracy here)
        scheduler.step(val_accuracy)

    return train_losses, val_accuracies, class_accuracies_list

# ==============================================================================
# 7. Save Results to Excel
# ==============================================================================
def save_results_to_excel(train_losses, val_accuracies, class_accuracies_list,
                          filename="training_results.xlsx"):
    data = {
        "Epoch": list(range(1, len(train_losses) + 1)),
        "Train Loss": train_losses,
        "Validation Accuracy": val_accuracies,
    }
    df = pd.DataFrame(data)

    # Convert class-wise accuracy dicts into aligned columns
    if class_accuracies_list:
        all_keys = sorted({k for d in class_accuracies_list for k in d.keys()})
        for k in all_keys:
            df[k] = [d.get(k, 0.0) for d in class_accuracies_list]

    df.to_excel(filename, index=False)
    print(f"Results saved to {filename}")

# ==============================================================================
# 8. Main Training Routine (Multiple Split Ratios)
# ==============================================================================
def train_model(data_dir, batch_size=16, epochs=20, learning_rate=1e-4):
    """
    Train the model under multiple train/validation split ratios:
    0.9/0.1 down to 0.4/0.6.
    Saves an Excel report for each split setting.
    """
    split_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]

    for ratio in split_ratios:
        print(f"\nTraining with train-validation split ratio: {int(ratio*10)}:{int((1-ratio)*10)}")

        dataset = DisasterDataset(root_dir=data_dir, transform=transform)
        train_size = int(ratio * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        model = DualSwinTransformerClassifier(num_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            [{"params": model.fusion.parameters()},
             {"params": model.classifier.parameters()}],
            lr=learning_rate
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

        train_losses, val_accuracies, class_accuracies_list = train_and_evaluate(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            epochs=epochs, patience=3
        )

        filename = f"training_results_{int(ratio*10)}_{int((1-ratio)*10)}.xlsx"
        save_results_to_excel(train_losses, val_accuracies, class_accuracies_list, filename)

# ==============================================================================
# 9. Entry Point
# ==============================================================================
if __name__ == "__main__":
    data_dir = r"/content/final_pair_label_classified_folders/final_pair_label_classified_folders"
    try:
        train_model(data_dir)
    except Exception as e:
        print(f"An error occurred during training: {e}")
