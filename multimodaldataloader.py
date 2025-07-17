import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 标签映射字典
LABEL_MAP = {
    'G1': 0, 'G2': 1, 'G3': 2, 'G4': 3, 'G5': 4,
    'G6': 5, 'G7': 6, 'G8': 7, 'G9': 8
}

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TappingSoundDataset(Dataset):
    def __init__(self, root_dir, target_length=10501, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录，其中包含 training/ 和 testing/ 文件夹
            target_length (int): 目标音频长度，默认为 10501
            transform (callable, optional): 一个可选的变换函数，用于音频数据的预处理
        """
        self.root_dir = root_dir
        self.target_length = target_length  # 设置目标长度
        self.transform = transform
        self.file_list = []

        # 遍历根目录中的所有文件（包括子文件夹中的文件）
        for subdir, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith('.wav'):
                    # 提取前缀作为类别标签
                    label_prefix = file_name[:2]  # 提取前两个字符，G1、G2等
                    label = LABEL_MAP.get(label_prefix)
                    if label is not None:
                        file_path = os.path.join(subdir, file_name)
                        self.file_list.append((file_path, label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # 确保音频为单声道，如果是多声道则取第一个声道
        if waveform.shape[0] > 1:
            waveform = waveform[0, :].unsqueeze(0)  # 保证维度为 [1, n]

        # 调整音频长度为 target_length
        if waveform.shape[1] > self.target_length:
            # 如果音频太长，裁剪
            waveform = waveform[:, :self.target_length]
        elif waveform.shape[1] < self.target_length:
            # 如果音频太短，填充
            pad_amount = self.target_length - waveform.shape[1]
            waveform = F.pad(waveform, (0, pad_amount))

        # 确保数据维度为 [10501, 1]
        waveform = waveform.t()  # 转置为 [10501, 1]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

class NoFlashImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 数据集的根目录，其中包含 training/ 和 testing/ 文件夹
            transform (callable, optional): 一个可选的变换函数，用于图像数据的预处理
        """
        self.root_dir = root_dir
        self.transform = transform
        self.file_list = []

        # 遍历根目录中的所有文件（包括子文件夹中的文件）
        for subdir, _, files in os.walk(root_dir):
            for file_name in files:
                if file_name.endswith('.jpg'):
                    # 提取前缀作为类别标签
                    label_prefix = file_name[:2]  # 提取前两个字符，G1、G2等
                    label = LABEL_MAP.get(label_prefix)
                    if label is not None:
                        file_path = os.path.join(subdir, file_name)
                        self.file_list.append((file_path, label))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path, label = self.file_list[idx]
        image = Image.open(file_path).convert('RGB')  # 打开并确保图像为RGB格式

        if self.transform:
            image = self.transform(image)

        return image, label



class MultimodalDataset(Dataset):
    def __init__(self, image_dataset, acoustics_dataset):
        """
        Args:
            image_dataset (Dataset): 图像数据集
            acoustics_dataset (Dataset): 音频数据集
        """
        assert len(image_dataset) == len(acoustics_dataset), "图像和音频数据集长度必须相同"
        self.image_dataset = image_dataset
        self.acoustics_dataset = acoustics_dataset

    def __len__(self):
        return len(self.image_dataset)

    def __getitem__(self, idx):
        # 从图像和音频数据集中获取数据
        image, img_label = self.image_dataset[idx]
        acoustics, acoustics_label = self.acoustics_dataset[idx]

        # 确保图像和音频的标签一致
        assert img_label == acoustics_label, f"图像标签 ({img_label}) 和音频标签 ({acoustics_label}) 不一致，索引：{idx}"


        return acoustics, image, img_label



"""
if __name__ == "__main__":
    # 创建图像和音频数据集
    image_dataset = NoFlashImageDataset("dataset/NoFlash/Training", transform=image_transforms)
    acoustics_dataset = TappingSoundDataset("dataset/Tapping/training", target_length=10501)

    # 创建多模态数据集，确保数据同步
    multimodal_dataset = MultimodalDataset(acoustics_dataset,image_dataset)

    # 创建多模态数据加载器
    multimodal_loader = DataLoader(multimodal_dataset, batch_size=32, shuffle=True)

    # 遍历数据加载器
    for i, (images, acoustics, labels) in enumerate(multimodal_loader):
        print(f"Batch {i}:")
        print(f"Images shape: {images.shape}")
        print(f"Acoustics shape: {acoustics.shape}")
        print(f"Labels: {labels}")
                
"""
