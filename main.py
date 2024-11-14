import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import math
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from multimodaldataloader import NoFlashImageDataset,TappingSoundDataset,MultimodalDataset
from model import MultimodalClassificationModel

# 定义图像预处理
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为224x224
    transforms.ToTensor(),  # 将图像转换为Tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准化
])

# 标签映射字典
LABEL_MAP = {
    'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4,
    'G6': 5, 'G7': 6, 'G8': 7, 'G9': 8
}


class Trainer:
    def __init__(self, model, train_loader, criterion, optimizer, device, results_dir):
        self.model = model
        self.train_loader = train_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.results_dir = results_dir

    def train(self, num_epochs):
        self.model.train()
        train_loss_file = os.path.join(self.results_dir, 'train_loss_accuracy.txt')

        with open(train_loss_file, 'w') as f:
            for epoch in range(num_epochs):
                running_loss = 0.0
                correct = 0
                total = 0

                for i, (images, acoustics, labels) in enumerate(self.train_loader):
                    images, acoustics, labels = images.to(self.device), acoustics.to(self.device), labels.to(
                        self.device)

                    # 前向传播
                    outputs = self.model(acoustics, images)
                    loss = self.criterion(outputs, labels)

                    # 反向传播和优化
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # 计算损失和准确率
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                avg_loss = running_loss / len(self.train_loader)

                # 保存每个 epoch 的损失和准确率
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%\n')

                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
class Evaluator:
    def __init__(self, model, test_loader, criterion, device, results_dir):
        self.model = model
        self.test_loader = test_loader
        self.criterion = criterion
        self.device = device
        self.results_dir = results_dir

    def evaluate(self):
        self.model.eval()
        all_labels = []
        all_preds = []
        total_loss = 0.0
        total = 0
        correct = 0

        with torch.no_grad():
            for i, (images, acoustics, labels) in enumerate(self.test_loader):
                images, acoustics, labels = images.to(self.device), acoustics.to(self.device), labels.to(
                    self.device)

                # 前向传播
                outputs = self.model(acoustics, images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                # 获取预测
                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())

                # 计算准确率
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # 计算各种指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')
        avg_loss = total_loss / len(self.test_loader)

        # 保存结果
        self.save_results(accuracy, precision, recall, f1, avg_loss, all_labels, all_preds)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")
        print(f"Loss: {avg_loss:.4f}")

    def save_results(self, accuracy, precision, recall, f1, avg_loss, labels, preds):
        result_file = os.path.join(self.results_dir, 'test_metrics.txt')

        with open(result_file, 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1-score: {f1:.4f}\n")
            f.write(f"Loss: {avg_loss:.4f}\n")

        # 生成混淆矩阵
        cm = confusion_matrix(labels, preds)
        cm_file = os.path.join(self.results_dir, 'confusion_matrix.png')
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig(cm_file)
        plt.close()

if __name__ == "__main__":
    # 初始化模型
    # 创建图像和音频数据集
    train_image_dataset = NoFlashImageDataset('dataset/NoFlash/Training', transform=image_transforms)
    train_acoustics_dataset = TappingSoundDataset('dataset/Tapping/Training', target_length=10501)

    test_image_dataset = NoFlashImageDataset('dataset/NoFlash/Testing', transform=image_transforms)
    test_acoustics_dataset = TappingSoundDataset('dataset/Tapping/Testing', target_length=10501)

    # 创建多模态数据集，确保数据同步且顺序为【声学，图像】
    train_multimodal_dataset = MultimodalDataset(train_acoustics_dataset, train_image_dataset)
    test_multimodal_dataset = MultimodalDataset(test_acoustics_dataset, test_image_dataset)

# 创建数据加载器
    train_multimodal_loader = DataLoader(train_multimodal_dataset, batch_size=32, shuffle=True)
    test_multimodal_loader = DataLoader(test_multimodal_dataset, batch_size=32, shuffle=False)



    results_dir = 'result'
    model = MultimodalClassificationModel(num_classes=9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)


    num_classes = 9
    batch_size = 16
    num_epochs = 200

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 实例化 Trainer 和 Evaluator
    trainer = Trainer(model, train_multimodal_loader, criterion, optimizer, device, results_dir)
    evaluator = Evaluator(model, test_multimodal_loader, criterion, device, results_dir)

    trainer.train(num_epochs=num_epochs)

    # 评估模型
    evaluator.evaluate()