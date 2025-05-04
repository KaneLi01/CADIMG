import os
from torch.utils.data import Dataset
from PIL import Image

class ControlNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        初始化数据集
        :param root_dir: 数据集根目录，包含多个样本文件夹
        :param transform: 图像变换（如数据增强）
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """
        加载数据集中的样本路径
        :return: 样本路径列表，每个样本是一个字典，包含初始图像、涂鸦图像、蒙版图像和结果图像的路径
        """
        samples = []
        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                sample = {
                    "initial": os.path.join(folder_path, "initial.png"),
                    "sketch": os.path.join(folder_path, "sketch.png"),
                    "mask": os.path.join(folder_path, "mask.png"),
                    "result": os.path.join(folder_path, "result.png"),
                }
                # 检查所有文件是否存在
                if all(os.path.exists(path) for path in sample.values()):
                    samples.append(sample)
        return samples

    def __len__(self):
        """
        返回数据集的样本数量
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        获取指定索引的样本
        :param idx: 样本索引
        :return: 包含初始图像、涂鸦图像、蒙版图像和结果图像的字典
        """
        sample = self.samples[idx]
        initial = Image.open(sample["initial"]).convert("RGB")
        sketch = Image.open(sample["sketch"]).convert("RGB")
        mask = Image.open(sample["mask"]).convert("L")  # 蒙版通常是灰度图
        result = Image.open(sample["result"]).convert("RGB")

        if self.transform:
            initial = self.transform(initial)
            sketch = self.transform(sketch)
            mask = self.transform(mask)
            result = self.transform(result)

        return {
            "initial": initial,
            "sketch": sketch,
            "mask": mask,
            "result": result,
        }