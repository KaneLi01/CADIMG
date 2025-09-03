import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms

class NormalSketchControlNetDataset(Dataset):
    def __init__(self, root_dir, mode='train', res=256):
        super().__init__()
        self.res = res
        assert mode in ['train', 'val', 'test'], "mode must be 'train', 'val' or 'test'"

        self.dir = os.path.join(root_dir, mode)
        self.base_dir = os.path.join(self.dir, "base_img")
        self.sketch_dir = os.path.join(self.dir, "sketch_img")
        self.target_dir = os.path.join(self.dir, "target_img")

        self.base_files = sorted(os.listdir(self.base_dir))
        self.sketch_files = sorted(os.listdir(self.sketch_dir))
        self.target_files = sorted(os.listdir(self.target_dir))

        self.files_len = len(self.base_files)
        
        # 检查数据集
        assert self.files_len == len(self.sketch_files) == len(self.target_files), \
            f"count error: base({self.files_len}), sketch({len(self.sketch_files)}), target({len(self.target_files)})"

        for b, s, t in zip(self.base_files, self.sketch_files, self.target_files):
            assert os.path.splitext(b)[0] == os.path.splitext(s)[0] == os.path.splitext(t)[0], \
                f"mismatch: {b} / {s} / {t}"
            
        print(f"total {self.files_len} imgs")

        self.normlize = transforms.Normalize([0.5], [0.5])
        self.transform = transforms.Compose([
                transforms.Resize((self.res, self.res)),
                transforms.ToTensor(),
            ])


    def __len__(self):
        return self.files_len

    def __getitem__(self, idx):
        filename = self.base_files[idx]

        base_path = os.path.join(self.base_dir, filename)  
        sketch_path = os.path.join(self.sketch_dir, filename)  
        target_path = os.path.join(self.target_dir, filename)  


        base_image = self.transform(Image.open(base_path).convert("RGB"))
        sketch_image = self.transform(Image.open(sketch_path).convert("RGB"))
        target_image = self.transform(Image.open(target_path).convert("RGB"))


        return {
            "base": base_image,  
            "sketch": self.normlize(sketch_image),     
            "target": self.normlize(target_image)    
        }
