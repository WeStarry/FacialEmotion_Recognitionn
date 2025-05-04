"""迁移学习文件
kaggle训练文件，将model.py中的模型定义写入内部方便直接运行"""



import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MiniXception(nn.Module):
    def __init__(self, num_classes=7, input_channels=1):
        super(MiniXception, self).__init__()

        # 添加图像预处理层
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((48, 48)),  # 统一调整图像大小为48x48
            nn.Conv2d(input_channels, 1, kernel_size=1)  # 如果是RGB图像，将其转换为灰度图
        )

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.25)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.25)

        self.residual1 = self._make_residual_block(64, 128)
        self.residual2 = self._make_residual_block(128, 256)
        self.residual3 = self._make_residual_block(256, 512)

        self.conv3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(512)
        self.dropout3 = nn.Dropout(0.5)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_residual_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            ChannelAttention(out_channels),  # 新增注意力
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            ChannelAttention(out_channels),  # 新增注意力
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Dropout(0.25)
        )

    def forward(self, x):
        # 预处理输入数据
        x = self.preprocess(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.residual1(x)
        x = self.residual2(x)
        x = self.residual3(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



# 设置全局参数
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.00005  # 降低学习率以便微调
IMG_SIZE = 48


# 表情标签映射
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
# CK+数据集文件夹名称映射到表情标签索引
folder_to_label = {
    'anger': 0,     # Angry
    'disgust': 1,   # Disgust
    'fear': 2,      # Fear
    'happiness': 3, # Happy
    'sadness': 4,   # Sad
    'surprise': 5,  # Surprise
    'neutral': 6    # Neutral
}
# 忽略 'contempt' 文件夹

class CKPlusDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # 读取图像并确保其为灰度图
        image = Image.open(img_path)
        if image.mode != 'L':  # 如果不是灰度图像
            image = image.convert('L')
            
        # 应用变换
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
            ])
            image = transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)


def load_ckplus_data(data_dir):
    """加载CK+数据集，排除'contempt'表情"""
    image_paths = []
    labels = []
    
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        
        # 跳过contempt文件夹
        if folder == 'contempt' or not os.path.isdir(folder_path):
            continue
            
        # 确保文件夹名称有效
        if folder not in folder_to_label:
            print(f"警告: 忽略未知文件夹 {folder}")
            continue
            
        label = folder_to_label[folder]
        
        # 遍历该表情文件夹中的所有图片
        for img_file in os.listdir(folder_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(folder_path, img_file)
                image_paths.append(img_path)
                labels.append(label)
    
    return image_paths, labels


def plot_confusion_matrix(cm, classes, title='混淆矩阵'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 绘制数量
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title('混淆矩阵 (数量)', fontsize=12)
    ax1.set_xlabel('预测标签', fontsize=10)
    ax1.set_ylabel('真实标签', fontsize=10)
    ax1.set_xticks(np.arange(len(classes)) + 0.5)
    ax1.set_yticks(np.arange(len(classes)) + 0.5)
    ax1.set_xticklabels(classes, rotation=45)
    ax1.set_yticklabels(classes, rotation=45)
    
    # 绘制归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2)
    ax2.set_title('混淆矩阵 (归一化)', fontsize=12)
    ax2.set_xlabel('预测标签', fontsize=10)
    ax2.set_ylabel('真实标签', fontsize=10)
    ax2.set_xticks(np.arange(len(classes)) + 0.5)
    ax2.set_yticks(np.arange(len(classes)) + 0.5)
    ax2.set_xticklabels(classes, rotation=45)
    ax2.set_yticklabels(classes, rotation=45)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    try:
        # 修改保存路径到Kaggle工作目录
        output_path = os.path.join('/kaggle/working/', f'{title}.png')
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"混淆矩阵已保存为: {output_path}")
    except Exception as e:
        print(f"警告: 保存混淆矩阵图失败: {str(e)}")
    finally:
        plt.close()


def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc='训练中', leave=False) as pbar:
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
                
                pbar.set_postfix({'损失': loss.item(), '准确率': correct/total})
            except Exception as e:
                print(f"训练批次出错: {str(e)}")
                continue

    return running_loss / len(train_loader), correct / total


def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_correct = [0] * len(emotion_labels)
    class_total = [0] * len(emotion_labels)

    with torch.no_grad():
        with tqdm(val_loader, desc='验证中', leave=False) as pbar:
            for inputs, labels in pbar:
                try:
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

                    pbar.set_postfix({'损失': loss.item(), '准确率': correct/total})

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except Exception as e:
                    print(f"验证批次出错: {str(e)}")
                    continue

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(len(emotion_labels)):
        if class_total[i] > 0:
            class_accuracies.append(class_correct[i] / class_total[i])
        else:
            class_accuracies.append(0.0)

    return running_loss / len(val_loader), correct / total, all_preds, all_labels, class_accuracies


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 修改数据集路径
    data_dir = '/kaggle/input/ck-48more/CK+48more'
    image_paths, labels = load_ckplus_data(data_dir)
    
    # 检查加载的数据
    print(f"加载了 {len(image_paths)} 张图片")
    label_counts = {emotion_labels[i]: labels.count(i) for i in range(len(emotion_labels))}
    print("数据分布:")
    for emotion, count in label_counts.items():
        print(f"  {emotion}: {count}")
    
    # 创建训练集、验证集和测试集（8:1:1分割）
    # 首先分出测试集
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # 再从剩余部分分出验证集
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels, 
        test_size=1/9,  # 这样验证集占总数据的1/10
        random_state=42, 
        stratify=train_val_labels
    )
    
    print(f"训练集: {len(train_paths)} 图片")
    print(f"验证集: {len(val_paths)} 图片")
    print(f"测试集: {len(test_paths)} 图片")
    
    # 定义数据增强和转换
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    
    # 创建数据集
    train_dataset = CKPlusDataset(train_paths, train_labels, transform=train_transform)
    val_dataset = CKPlusDataset(val_paths, val_labels, transform=val_transform)
    test_dataset = CKPlusDataset(test_paths, test_labels, transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 加载预训练模型 - 修改模型路径
    model = MiniXception(num_classes=7, input_channels=1).to(device)
    pretrained_model_path = '/kaggle/input/best_model_fold0.pth/pytorch/default/1/best_model_fold0.pth'
    model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
    print(f"成功加载预训练模型 {pretrained_model_path}")
    
    # 冻结部分参数（只微调后面几层）
    # 冻结前面的卷积层
    for name, param in model.named_parameters():
        if 'conv1' in name or 'bn1' in name or 'conv2' in name or 'bn2' in name:
            param.requires_grad = False
    
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # 训练模型
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels, class_accuracies = validate_model(
            model, val_loader, criterion, device)
        scheduler.step(val_loss)
        
        print(f"周期 {epoch + 1}/{EPOCHS} - "
              f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}, "
              f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            print("\n每个类别的准确率:")
            for i, emotion in enumerate(emotion_labels):
                print(f"{emotion}: {class_accuracies[i]:.4f}")
            
            # 计算并绘制混淆矩阵
            cm = confusion_matrix(val_labels, val_preds)
            plot_confusion_matrix(cm, emotion_labels, f"验证集混淆矩阵_周期{epoch+1}")
        
        # 保存最佳模型 - 修改保存路径
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            output_model_path = '/kaggle/working/ckplus_best_model.pth'
            torch.save(model.state_dict(), output_model_path)
            print(f"新的最佳模型已保存，验证准确率: {best_val_acc:.4f}，保存路径: {output_model_path}")
    
    # 最终测试评估
    print("\n加载最佳模型进行测试...")
    model.load_state_dict(torch.load('/kaggle/working/ckplus_best_model.pth'))
    test_loss, test_acc, test_preds, test_labels, test_class_accuracies = validate_model(
        model, test_loader, criterion, device)
    
    print(f"\n测试结果 - 准确率: {test_acc:.4f}")
    print("\n每个类别的测试准确率:")
    for i, emotion in enumerate(emotion_labels):
        print(f"{emotion}: {test_class_accuracies[i]:.4f}")
    
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=emotion_labels))
    
    # 绘制测试集混淆矩阵
    test_cm = confusion_matrix(test_labels, test_preds)
    plot_confusion_matrix(test_cm, emotion_labels, "测试集混淆矩阵")


if __name__ == '__main__':
    main() 