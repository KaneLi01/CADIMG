import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SketchControlNetDataset(Dataset):
    def __init__(self, root_dir):

        super().__init__()
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "init_img")
        self.stroke_dir = os.path.join(root_dir, "stroke_img")
        self.target_dir = os.path.join(root_dir, "result_img")
        self.mask_dir = os.path.join(root_dir, "mask_img")
        # 获取所有图片名字
        self.image_filenames = os.listdir(self.img_dir)
        
        # 定义图像
        self.normlize_1 = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
                transforms.Resize((128,128)),
                transforms.ToTensor()
            ])


    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        filename = self.image_filenames[idx]

        img_path = os.path.join(self.img_dir, filename)  # 作为stable diffusion的输入
        stroke_path = os.path.join(self.stroke_dir, filename)  # 手绘灰度图
        target_path = os.path.join(self.target_dir, filename)  # 匹配的目标图
        mask_path = os.path.join(self.mask_dir, filename)  # mask

        img_image = self.transform(Image.open(img_path).convert("RGB"))
        stroke_image = self.transform(Image.open(stroke_path).convert("RGB"))
        target_image = self.transform(Image.open(target_path).convert("RGB"))
        mask_image = self.transform(Image.open(mask_path).convert("L"))

        return {
            "original": img_image,  
            "sketch": self.normlize_1(stroke_image),     
            "target": self.normlize_1(target_image),      
            "mask": mask_image         
        }
