# ==============================================================================
# ConvNeXt Tiny – Progressive Train/Validation Split Experiment with Grad-CAM
# ==============================================================================

import torch
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch import nn, optim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from pytorch_grad_cam import ScoreCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# -----------------------------
# 1. Device & Dataset Path
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_dir = "/content/final_label_image_post/final_label_image_post"

# -----------------------------
# 2. Data Augmentation & Preprocessing
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = ImageFolder(root=data_dir, transform=transform)
class_names = dataset.classes  # Retrieve class names

# -----------------------------
# 3. Train–Validation Splits
#    Ratios vary from 9:1 down to 4:6
# -----------------------------
split_ratios = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
batch_size = 16
epochs = 20
patience = 3

for ratio in split_ratios:
    print(f"\n🔹 Training with train-validation split ratio: {ratio:.1f}")

    train_size = int(ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # -----------------------------
    # 4. ConvNeXt Tiny Classifier Definition
    # -----------------------------
    class ConvNeXtClassifier(nn.Module):
        def __init__(self, num_classes=3):
            super(ConvNeXtClassifier, self).__init__()
            self.convnext_model = timm.create_model("convnext_tiny", pretrained=True, num_classes=0)
            self.fc = nn.Linear(self.convnext_model.num_features, num_classes)

        def forward(self, x):
            features = self.convnext_model(x)
            output = self.fc(features)
            return output

    model = ConvNeXtClassifier(num_classes=3).to(device)

    # -----------------------------
    # 5. Training & Evaluation
    # -----------------------------
    def train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, patience=3):
        best_accuracy = 0.0
        patience_counter = 0
        results = []

        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct_train, total_train = 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct_train / total_train
            scheduler.step()

            # ----- Validation -----
            model.eval()
            correct_val, total_val = 0, 0
            all_preds, all_labels = [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            val_accuracy = 100 * correct_val / total_val
            precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) * 100
            mcc = matthews_corrcoef(all_labels, all_preds) * 100

            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
                  f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%, "
                  f"P: {precision:.2f}, R: {recall:.2f}, F1: {f1:.2f}, MCC: {mcc:.2f}")

            results.append({
                'Epoch': epoch + 1,
                'Train Loss': train_loss,
                'Train Accuracy': train_accuracy,
                'Validation Accuracy': val_accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1 Score': f1,
                'MCC': mcc
            })

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        return results

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    results = train_and_evaluate(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                 epochs=epochs, patience=patience)

    # -----------------------------
    # 6. Save Model & Results
    # -----------------------------
    model_save_path = f"fine_tuned_convnext_tiny_{int(ratio*10)}_{int((1-ratio)*10)}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved at {model_save_path}")

    filename = f"training_results_{int(ratio*10)}_{int((1-ratio)*10)}.xlsx"
    pd.DataFrame(results).to_excel(filename, index=False)
    print(f"📊 Results saved to {filename}")

# -----------------------------
# 7. Grad-CAM Heatmap Visualization
# -----------------------------
def generate_heatmap(model, target_layer, image_path):
    model.eval()
    orig_image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(orig_image).unsqueeze(0).to(device)

    cam = ScoreCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    heatmap = cv2.resize(grayscale_cam, orig_image.size)
    result_cam = show_cam_on_image(np.array(orig_image) / 255.0, heatmap, use_rgb=True)

    plt.imshow(result_cam)
    plt.axis("off")
    plt.show()

example_image_path = "/content/389812700865067_2024.png"
generate_heatmap(model, model.convnext_model.stages[3], example_image_path)
