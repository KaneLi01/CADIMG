"""
模型设置器 - 负责模型初始化、Pipeline构建和优化器设置
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from diffusers import StableDiffusionControlNetPipeline

from models.imgedit.diffusion import Diffusion_Models


class ModelSetup:
    """模型设置器"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.models = None
        self.train_pipe = None
        self.optimizer = None
        self.loss_fn = None
        
    def initialize_models(self) -> None:
        """初始化扩散模型"""
        print("Initializing diffusion models...")
        self.models = Diffusion_Models(self.config)

    def create_training_pipeline(self) -> StableDiffusionControlNetPipeline:
        """
        创建训练Pipeline
        
        Returns:
            训练用的Pipeline
        """
        if self.models is None:
            raise ValueError("Models must be initialized first")
            
        print("Creating training pipeline...")
        
        self.train_pipe = StableDiffusionControlNetPipeline(
            vae=self.models.vae,
            text_encoder=None,  # 不使用文本编码器，而是使用CLIP图像编码
            tokenizer=None,
            unet=self.models.unet,
            controlnet=self.models.controlnet,
            scheduler=self.models.scheduler,
            safety_checker=None,
            feature_extractor=None
        ).to(self.device)
        
        # 冻结UNet参数 - 只训练ControlNet
        self._freeze_unet_parameters()
        
        print("✓ Training pipeline created")
        return self.train_pipe
    
    def _freeze_unet_parameters(self) -> None:
        """冻结UNet参数"""
        frozen_params = 0
        total_params = 0
        
        for param in self.train_pipe.unet.parameters():
            total_params += 1
            param.requires_grad = False
            frozen_params += 1
            
        print(f"✓ UNet parameters frozen: {frozen_params}/{total_params}")
        
    def setup_optimizer(self) -> optim.Optimizer:
        """
        设置优化器
        
        Returns:
            配置好的优化器
        """
        if self.train_pipe is None:
            raise ValueError("Training pipeline must be created first")
            
        print("Setting up optimizer...")
        
        # 只优化ControlNet参数
        trainable_params = list(self.train_pipe.controlnet.parameters())
        trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        
        print(f"✓ Optimizer setup complete. Trainable parameters: {trainable_count:,}")
        return self.optimizer
    
    def setup_loss_function(self) -> nn.Module:
        """
        设置损失函数
        
        Returns:
            损失函数
        """
        print("Setting up loss function...")
        
        # 使用MSE损失进行噪声预测
        self.loss_fn1 = nn.MSELoss()
        self.loss_fn2 = nn.MSELoss()
        
        return self.loss_fn1, self.loss_fn2
    
    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            模型信息字典
        """
        if self.train_pipe is None:
            return {"error": "Models not initialized"}
            
        info = {}
        
        # ControlNet参数统计
        controlnet_params = sum(
            p.numel() for p in self.train_pipe.controlnet.parameters()
        )
        controlnet_trainable = sum(
            p.numel() for p in self.train_pipe.controlnet.parameters() 
            if p.requires_grad
        )
        
        # UNet参数统计  
        unet_params = sum(
            p.numel() for p in self.train_pipe.unet.parameters()
        )
        unet_trainable = sum(
            p.numel() for p in self.train_pipe.unet.parameters() 
            if p.requires_grad
        )
        
        info.update({
            'controlnet_params': controlnet_params,
            'controlnet_trainable': controlnet_trainable,
            'unet_params': unet_params,
            'unet_trainable': unet_trainable,
            'total_params': controlnet_params + unet_params,
            'total_trainable': controlnet_trainable + unet_trainable,
            'device': str(self.device),
            'optimizer': type(self.optimizer).__name__ if self.optimizer else None,
            'loss_function': type(self.loss_fn).__name__ if self.loss_fn else None,
        })
        
        return info
    
    def print_model_summary(self) -> None:
        """打印模型摘要"""
        info = self.get_model_info()
        
        if 'error' in info:
            print(f"❌ {info['error']}")
            return
            
        print("\n" + "="*50)
        print("🚀 MODEL SUMMARY")
        print("="*50)
        print(f"📊 Total Parameters: {info['total_params']:,}")
        print(f"🎯 Trainable Parameters: {info['total_trainable']:,}")
        print(f"📈 Training Ratio: {info['total_trainable']/info['total_params']*100:.2f}%")
        print("\n📋 Component Details:")
        print(f"  • ControlNet: {info['controlnet_params']:,} ({info['controlnet_trainable']:,} trainable)")
        print(f"  • UNet: {info['unet_params']:,} ({info['unet_trainable']:,} trainable)")
        print(f"\n🔧 Configuration:")
        print(f"  • Device: {info['device']}")
        print(f"  • Optimizer: {info['optimizer']}")
        print(f"  • Loss Function: {info['loss_function']}")
        print("="*50 + "\n")
    
    def setup_all(self) -> Tuple[StableDiffusionControlNetPipeline, optim.Optimizer, nn.Module]:
        """
        一次性设置所有组件
        
        Returns:
            (pipeline, optimizer, loss_function)
        """
        print("🚀 Starting model setup...")
        
        # 按顺序初始化所有组件
        self.initialize_models()
        pipeline = self.create_training_pipeline()
        optimizer = self.setup_optimizer()
        loss_fn = self.setup_loss_function()
        
        # 打印摘要
        self.print_model_summary()
        
        print("✅ Model setup complete!\n")
        
        return pipeline, optimizer, loss_fn
    
    def validate_setup(self) -> bool:
        """
        验证模型设置是否正确
        
        Returns:
            设置是否有效
        """
        if self.models is None:
            print("❌ Models not initialized")
            return False
            
        if self.train_pipe is None:
            print("❌ Training pipeline not created")
            return False
            
        if self.optimizer is None:
            print("❌ Optimizer not setup")
            return False
            
        if self.loss_fn is None:
            print("❌ Loss function not setup")
            return False
            
        # 检查UNet是否被正确冻结
        unet_trainable = any(p.requires_grad for p in self.train_pipe.unet.parameters())
        if unet_trainable:
            print("⚠️  Warning: UNet parameters are not frozen")
            return False
            
        # 检查ControlNet是否可训练
        controlnet_trainable = any(p.requires_grad for p in self.train_pipe.controlnet.parameters())
        if not controlnet_trainable:
            print("❌ ControlNet parameters are not trainable")
            return False
            
        print("✅ Model setup validation passed")
        return True


def create_model_setup(config) -> ModelSetup:
    """
    工厂函数：创建模型设置器
    
    Args:
        config: 配置对象
        
    Returns:
        模型设置器实例
    """
    return ModelSetup(config)