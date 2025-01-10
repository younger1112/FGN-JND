import os
import cv2
import numpy as np
from torchvision import transforms
from dataset import CustomDataset

# 定义数据集路径
image_folder = 'images/'
saliency_folder = 'generated_saliency_masks/'

# 读取图像和显著性图
image_filenames = os.listdir(image_folder)
saliency_filenames = os.listdir(saliency_folder)

# 检查图像和显著性图是否一一对应
assert len(image_filenames) == len(saliency_filenames)

# 创建数据集
dataset = []
for image_name, saliency_name in zip(image_filenames, saliency_filenames):
    image_path = os.path.join(image_folder, image_name)
    saliency_path = os.path.join(saliency_folder, saliency_name)
    
    # 读取图像和显著性图
    image = cv2.imread(image_path)
    saliency_map = cv2.imread(saliency_path, cv2.IMREAD_GRAYSCALE)
    
    # 预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((640, 480)),  # 假设模型输入大小为 224x224
    ])
    image = transform(image)
    saliency_map = transform(saliency_map)
    
    # 添加到数据集
    dataset.append((image, saliency_map))

# 创建自定义数据集对象
custom_dataset = CustomDataset(dataset)
