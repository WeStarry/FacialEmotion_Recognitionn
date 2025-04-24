import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from model import MiniXception
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

BATCH_SIZE = 32
EPOCHS = 50
LR = 0.0001
K_FOLDS = 5
IMG_SIZE = 48


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

class RAFDBDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Grayscale(1),  # 转换为灰度图
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像大小为48x48
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]) if transform else transforms.Compose([
            transforms.Grayscale(1),  # 转换为灰度图
            transforms.Resize((IMG_SIZE, IMG_SIZE)),  # 调整图像大小为48x48
            transforms.ToTensor(),
        ])

        self.classes = ['angry', 'disgust', 'fear', 'happy', 'natural', 'sad', 'surprise']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), self.class_to_idx[class_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label).long()


def load_data():
    # 加载FER2013数据集
    df = pd.read_csv('fer2013.csv')
    pixels = df['pixels'].values
    emotions = df['emotion'].values
    
    # 加载RAF-DB数据集
    rafdb_train = RAFDBDataset(os.path.join('RAF-DB', 'train'))
    rafdb_test = RAFDBDataset(os.path.join('RAF-DB', 'test'))
    
    return pixels, emotions, rafdb_train, rafdb_test


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc='Training', leave=False) as pbar:
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    return running_loss / len(train_loader), correct / total


def plot_confusion_matrix(cm, classes, fold, epoch):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix (Fold {fold+1}, Epoch {epoch+1})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.yticks(np.arange(len(classes)) + 0.5, classes, rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_fold{fold}_epoch{epoch+1}.png')
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
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # 计算每个类别的准确率
                for i in range(len(labels)):
                    label = labels[i]
                    class_total[label] += 1
                    if predicted[i] == label:
                        class_correct[label] += 1

                pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(emotion_labels)):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return running_loss / len(val_loader), correct / total, all_preds, all_labels, class_accuracies


# 实现Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 权重系数
        self.gamma = gamma  # 聚焦参数，增加难分类样本的权重
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)  # 预测概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss  # 应用聚焦因子
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    pixels, emotions, rafdb_train, rafdb_test = load_data()
    class_weights = compute_class_weight('balanced', classes=np.unique(emotions), y=emotions)
    
    # 增加Fear类别的权重
    fear_idx = 2  # Fear类别的索引
    class_weights[fear_idx] *= 1.2  # 增加Fear类别的权重（从1.5降低到1.2）
    
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # 定义情绪标签
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    kfold = KFold(n_splits=K_FOLDS, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(pixels)):
        print(f"\nFold {fold + 1}/{K_FOLDS}")

        train_pixels, val_pixels = pixels[train_idx], pixels[val_idx]
        train_emotions, val_emotions = emotions[train_idx], emotions[val_idx]

        # 创建FER2013数据集
        fer_train_dataset = FERDataset(train_pixels, train_emotions, transform=True)
        fer_val_dataset = FERDataset(val_pixels, val_emotions)
        
        # 将RAF-DB训练集按比例分割为训练集和验证集
        rafdb_train_size = int(0.8 * len(rafdb_train))
        rafdb_val_size = len(rafdb_train) - rafdb_train_size
        rafdb_train_dataset, rafdb_val_dataset = torch.utils.data.random_split(
            rafdb_train, [rafdb_train_size, rafdb_val_size])

        # 合并两个数据集
        train_dataset = ConcatDataset([fer_train_dataset, rafdb_train_dataset])
        val_dataset = ConcatDataset([fer_val_dataset, rafdb_val_dataset])
        
        # 为训练集创建WeightedRandomSampler
        # 获取所有训练样本的标签
        all_train_labels = []
        for i in range(len(train_dataset)):
            _, label = train_dataset[i]
            all_train_labels.append(label.item())
        
        # 计算每个类别的样本权重
        label_to_count = np.bincount(all_train_labels)
        weight_per_class = 1. / label_to_count
        # 增加Fear类别的采样权重
        weight_per_class[2] *= 1.3  # Fear类别的索引为2（从2.0降低到1.3）
        
        # 为每个样本分配权重
        weights = [weight_per_class[label] for label in all_train_labels]
        sampler = WeightedRandomSampler(weights, len(weights))
        
        # 使用WeightedRandomSampler创建DataLoader
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

        model = MiniXception(input_channels=1).to(device)
        # 使用Focal Loss替代CrossEntropyLoss
        criterion = FocalLoss(alpha=class_weights, gamma=1.5)  # gamma从2.0降低到1.5
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_acc = 0.0

        for epoch in range(EPOCHS):
            train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, val_preds, val_labels, class_accuracies = validate_model(model, val_loader, criterion, device, emotion_labels)
            scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 每5个批次输出各个表情的预测准确率和混淆矩阵
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

        model.load_state_dict(torch.load(f'best_model_fold{fold}.pth'))
        _, _, final_preds, final_labels, _ = validate_model(model, val_loader, criterion, device, emotion_labels)

        print("\nClassification Report:")
        print(classification_report(final_labels, final_preds))

        # 绘制最终的混淆矩阵
        final_cm = confusion_matrix(final_labels, final_preds)
        plot_confusion_matrix(final_cm, emotion_labels, fold, EPOCHS-1)


if __name__ == '__main__':
    main()