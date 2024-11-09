import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

class WeldDataset(Dataset):
    def __init__(self, ann_file, transform=None):
        """
        初始化WeldDataset
        :param ann_file: 数据集的标注文件
        :param transform: 应用于图像的预处理和转换
        """
        self.ann_file = ann_file
        self.transform = transform
        self.init()

    # def init(self):
    #     self.im_names = []
    #     self.targets = []
    #     with open(self.ann_file, 'r') as f:
    #         lines = f.readlines()
    #         for line in lines:
    #             data = line.strip().split('=')
    #             self.im_names.append(data[0])# 图像路径
    #             self.targets.append(int(data[1]))# 标签，这里的标签表示图像所在的文件夹编号
    def init(self):
        self.im_names = []
        self.targets = []
        with open(self.ann_file, 'r') as f:
            lines = f.readlines()
            data = [line.strip().split('=') for line in lines]
        random.shuffle(data)# 打乱数据

        #选取前10000个数据
        selected_data = data[:10000]

        for item in selected_data:
            self.im_names.append(item[0])  # 图像路径
            self.targets.append(int(item[1]))  # 标签，这里的标签表示图像所在的文件夹编号
        

    def __len__(self):
        """
        返回数据集中的图像数量
        """
        return len(self.im_names)

    def __getitem__(self, index):
        """
        获取单个图像和标签
        """
        im_name = self.im_names[index]
        target = self.targets[index]
        image = Image.open(im_name).convert('RGB')

        if image is None:# 如果图像为空
            print(im_name)

        img = self.transform(image)

        return img, img
    
def train_loader(data_list, batch_size, num_workers, shuffle=True):
    # 图像增强
    augmentation = [
        transforms.Resize((512,512)),
        transforms.RandomResizedCrop(512, scale=(0.2, 1.)),# 随机裁剪
        transforms.RandomHorizontalFlip(),# 随机水平翻转
        transforms.ToTensor(),
    ]

    train_trans = transforms.Compose(augmentation)

    train_dataset = WeldDataset(data_list, transform=train_trans)

    train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=True)
    return train_loader

def test_loader(data_list, batch_size, num_workers, shuffle=False):

    test_trans = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    test_dataset = WeldDataset(data_list, transform=test_trans)

    test_loader = torch.utils.data.DataLoader(
                    test_dataset,
                    shuffle=shuffle,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    pin_memory=True,
                    drop_last=False)
    
    return test_loader

        
# 定义数据集类
class ImageDatasetSearch(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.images = []
        self.image_paths = []
        self.classes = []
        self.objects = []

        # 字典序列, 使用 int 映射 类名
        self.class_to_idx = {}
        # 字典序列, 使用 int 映射 物体名称
        self.object_to_idx = {}
        self.object_to_class = {}
        
        # 遍历数据集目录
        for class_name in os.listdir(root_dir):
            # class_name: 类名 比如：0 1 2 3 10 101
            class_path = os.path.join(root_dir, class_name)

            # 实现 int 到类名的映射
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            # class_idx: 得到类名对应的int
            class_idx = self.class_to_idx[class_name]
            
            # 遍历某个分类下的所有物体，比如 101 下的 SHII005-Z022-W-R-XF-01、SHII005-Z022-W-R-XF-01
            for object_name in os.listdir(class_path):
                # object_name: 物体名称
                object_path = os.path.join(class_path, object_name)
                if not os.path.isdir(object_path):
                    continue
                # 实现int到物体名称的映射
                if object_name not in self.object_to_idx:
                    self.object_to_idx[object_name] = len(self.object_to_idx)
                # 得到物体名称对应的id
                object_idx = self.object_to_idx[object_name]
                # 物体到分类的对应关系
                self.object_to_class[object_idx] = class_idx
                
                ## 遍历物体下的所有 切片图像(因为将一个物体按照512*512切成了多份, 因此一个物体会有多个图像)
                for image_name in os.listdir(object_path):
                    if image_name.endswith('.png'):
                        # image_path: 切片图像的路径
                        image_path = os.path.join(object_path, image_name)
                        
                        
                        # 将切片图像的路径、所属类别对应的int、所属物体对应的int 
                        # 添加到 image_paths、classes、objects中
                        self.image_paths.append(image_path)
                        self.classes.append(class_idx)
                        self.objects.append(object_idx)
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        class_idx = self.classes[idx]
        object_idx = self.objects[idx]
        return image, image_path, class_idx, object_idx



