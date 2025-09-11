"""
数据处理器 - 负责图像预处理和批次数据处理
"""
from pathlib import Path
from typing import Dict, Tuple
import torch
from torchvision import transforms
from PIL import Image

from utils import img_utils  


class DataProcessor:
    """数据处理器"""
    
    def __init__(self, device: torch.device, resolution: int = 512):
        self.device = device
        self.resolution = resolution
        
        # 预定义变换
        self.image_transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
        ])
        
        self.sketch_normalize = transforms.Normalize([0.5], [0.5])
    
    def preprocess_image(self, image_path: Path) -> torch.Tensor:
        """
        预处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像张量
        """
        image = Image.open(image_path).convert("RGB")
        return self.image_transform(image).unsqueeze(0).to(self.device)
    
    def process_training_batch(
        self, 
        batch: Dict[str, torch.Tensor],
        models,
        target_dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        处理训练批次数据
        
        Args:
            batch: 批次数据
            models: 模型对象
            target_dtype: 目标数据类型
            
        Returns:
            (image_embeds, latents, noise, timesteps)
        """
        # 提取批次数据
        base = batch["base"].to(self.device, dtype=target_dtype)
        sketch = batch["sketch"].to(self.device, dtype=target_dtype)
        target = batch["target"].to(self.device, dtype=target_dtype)
        
        # 生成CLIP嵌入
        image_embeds = self._generate_clip_embeddings(base, models)
        
        # 编码目标图像为latents
        latents = self._encode_to_latents(target, models)
        
        # 生成噪声和时间步
        noise = torch.randn_like(latents)
        timesteps = self._sample_timesteps(latents.shape[0], models)
        
        return {
            "image_embeds": image_embeds,
            "latents": latents,
            "noise": noise,
            "timesteps": timesteps,
            "sketch": sketch,
        }
    
    def _generate_clip_embeddings(self, images: torch.Tensor, models) -> torch.Tensor:
        """生成CLIP图像嵌入"""
        with torch.no_grad():
            clip_input = models.clip_processor(
                images=images, return_tensors="pt"
            ).pixel_values.to(self.device)
            
            image_embeds = models.clip_model(clip_input).last_hidden_state
            
        return image_embeds
    
    def _encode_to_latents(self, images: torch.Tensor, models) -> torch.Tensor:
        """将图像编码为潜在表示"""
        with torch.no_grad():
            latents = models.vae.encode(images).latent_dist.sample() * 0.18215
        return latents
    
    def _sample_timesteps(self, batch_size: int, models) -> torch.Tensor:
        """采样时间步"""
        timesteps = torch.randint(
            0, models.scheduler.config.num_train_timesteps,
            (batch_size,), device=self.device
        ).long()
        return timesteps
    
    def process_test_images(
        self, 
        img_path: Path, 
        sketch_path: Path,
        models
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        处理测试图像
        
        Args:
            img_path: 基础图像路径
            sketch_path: 草图路径
            models: 模型对象
            
        Returns:
            (prompt_embeds, pooled_embeds, sketch_tensor)
        """
        # 预处理图像
        test_image = self.preprocess_image(img_path)
        test_sketch = self.sketch_normalize(self.preprocess_image(sketch_path))
        
        with torch.no_grad():
            # 生成CLIP嵌入
            test_clip_input = models.clip_processor(
                images=test_image, return_tensors="pt"
            ).pixel_values.to(self.device)
            test_embeds = models.clip_model(test_clip_input).last_hidden_state
            
            # 准备prompt嵌入
            # prompt_embeds = torch.cat([test_embeds, test_embeds], dim=0)
            pooled_embeds = test_embeds.mean(dim=1)
            
        return {
            'prompt_embeds': test_embeds, 
            'pooled_embeds': pooled_embeds, 
            'test_sketch': test_sketch
        }
    
    def create_input_visualization(self, img_path: Path, sketch_path: Path) -> Image.Image:
        """
        创建输入图像的可视化（将base和sketch并排显示）
        
        Args:
            img_path: 基础图像路径
            sketch_path: 草图路径
            
        Returns:
            合并后的图像
        """
        
        base_img = Image.open(img_path)
        sketch_img = Image.open(sketch_path)
        
        # 使用工具函数合并图像
        combined = img_utils.stack_imgs(base_img, sketch_img, mode='ew')
        return combined.resize((self.resolution, self.resolution), Image.BILINEAR)
    
    def add_noise_to_latents(
        self, 
        latents: torch.Tensor, 
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        scheduler
    ) -> torch.Tensor:
        """
        向潜在表示添加噪声
        
        Args:
            latents: 原始潜在表示
            noise: 噪声张量
            timesteps: 时间步
            scheduler: 调度器
            
        Returns:
            加噪后的潜在表示
        """
        return scheduler.add_noise(latents, noise, timesteps)


class DataProcessorConfig:
    """数据处理器配置"""
    
    def __init__(self, resolution: int = 512, normalize_sketch: bool = True):
        self.resolution = resolution
        self.normalize_sketch = normalize_sketch


# 工具函数
def get_image_stats(image_tensor: torch.Tensor) -> Dict[str, float]:
    """获取图像张量的统计信息"""
    return {
        'mean': image_tensor.mean().item(),
        'std': image_tensor.std().item(),
        'min': image_tensor.min().item(),
        'max': image_tensor.max().item(),
        'shape': list(image_tensor.shape)
    }


def normalize_tensor(tensor: torch.Tensor, mean: float = 0.5, std: float = 0.5) -> torch.Tensor:
    """标准化张量"""
    return (tensor - mean) / std