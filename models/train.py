
#训练策略文件

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from model import MiniXception
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局参数
BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
K_FOLDS = 5
IMG_SIZE = 48

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class FERDataset(Dataset):
    def __init__(self, pixels, labels, transform=None):
        self.pixels = pixels
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]) if transform else None

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        pixel = self.pixels[idx]
        label = self.labels[idx]

        pixel = np.array(pixel.split(), dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE, 1)
        pixel = pixel / 255.0

        if self.transform:
            pixel = self.transform(pixel)
        else:
            pixel = torch.from_numpy(pixel).permute(2, 0, 1).float()

        label = torch.tensor(label).long()
        return pixel, label


def load_data():
    df = pd.read_csv('fer2013.csv')
    pixels = df['pixels'].values
    emotions = df['emotion'].values
    return pixels, emotions


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc='Training', leave=False) as pbar:
        for inputs, labels in pbar:
            try:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # 添加梯度裁剪防止梯度爆炸
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
            except Exception as e:
                print(f"训练批次出错: {str(e)}")
                continue

    return running_loss / len(train_loader), correct / total


def plot_confusion_matrix(cm, classes, fold, epoch):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Plot counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('Confusion Matrix (Count)', fontsize=12)
    ax1.set_xlabel('Predicted Label', fontsize=10)
    ax1.set_ylabel('True Label', fontsize=10)
    ax1.set_xticks(np.arange(len(classes)) + 0.5)
    ax1.set_yticks(np.arange(len(classes)) + 0.5)
    ax1.set_xticklabels(classes, rotation=45)
    ax1.set_yticklabels(classes, rotation=45)
    
    # Plot normalized
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=12)
    ax2.set_xlabel('Predicted Label', fontsize=10)
    ax2.set_ylabel('True Label', fontsize=10)
    ax2.set_xticks(np.arange(len(classes)) + 0.5)
    ax2.set_yticks(np.arange(len(classes)) + 0.5)
    ax2.set_xticklabels(classes, rotation=45)
    ax2.set_yticklabels(classes, rotation=45)
    
    plt.suptitle(f'Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})', fontsize=16)
    plt.tight_layout()
    
    try:
        plt.savefig(f'confusion_matrix_fold{fold}_epoch{epoch+1}.png', 
                   bbox_inches='tight', dpi=300)
    except Exception as e:
        print(f"Warning: Failed to save confusion matrix plot: {str(e)}")
    finally:
        plt.close()


def validate_model(model, val_loader, criterion, device, emotion_labels):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * len(emotion_labels)
    class_total = [0] * len(emotion_labels)

    with torch.no_grad():
        with tqdm(val_loader, desc='Validating', leave=False) as pbar:
            for inputs, labels in pbar:
                try:
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    # Calculate per-class accuracy
                    for i in range(len(labels)):
                        label = labels[i]
                        class_total[label] += 1
                        if predicted[i] == label:
                            class_correct[label] += 1

                    pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"验证批次出错: {str(e)}")
                    continue

    # Calculate per-class accuracies
    class_accuracies = []
    for i in range(len(emotion_labels)):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return running_loss / len(val_loader), correct / total, all_preds, all_labels, class_accuracies


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pixels, emotions = load_data()
    class_weights = compute_class_weight('balanced', classes=np.unique(emotions), y=emotions)
    
    # Increase weight for Fear class
    fear_idx = 2
    class_weights[fear_idx] *= 1.2
    
    class_weights = torch.from_numpy(class_weights).to(torch.float32).to(device)
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(pixels)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        train_pixels, val_pixels = pixels[train_idx], pixels[val_idx]
        train_emotions, val_emotions = emotions[train_idx], emotions[val_idx]

        train_dataset = FERDataset(train_pixels, train_emotions, transform=True)
        val_dataset = FERDataset(val_pixels, val_emotions)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = MiniXception(input_channels=1).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels, class_accuracies = validate_model(
                model, val_loader, criterion, device, emotion_labels)
            scheduler.step(val_loss)

            print(f"Epoch {epoch + 1}/{EPOCHS} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if (epoch + 1) % 5 == 0:
                print("\nPer-class accuracies:")
                for i, emotion in enumerate(emotion_labels):
                    print(f"{emotion}: {class_accuracies[i]:.4f}")
                
                cm = confusion_matrix(val_labels, val_preds)
                plot_confusion_matrix(cm, emotion_labels, fold, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), f'best_model_fold{fold}.pth')
                print(f"New best model saved with val acc: {best_val_acc:.4f}")

        # Final evaluation
        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        _, _, final_preds, final_labels, _ = validate_model(
            model, val_loader, criterion, device, emotion_labels)

        print("\nClassification Report:")
        print(classification_report(final_labels, final_preds))

        final_cm = confusion_matrix(final_labels, final_preds)
        plot_confusion_matrix(final_cm, emotion_labels, fold, EPOCHS-1)


if __name__ == '__main__':
    main()