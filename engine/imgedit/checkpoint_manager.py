"""
检查点管理器 - 负责模型检查点的保存和加载
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
import torch
import torch.nn as nn


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, device: torch.device):
        self.device = device
    
    def save_checkpoint(
        self, 
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        save_path: Path,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        保存检查点
        
        Args:
            model: 要保存的模型
            optimizer: 优化器
            epoch: 当前epoch
            save_path: 保存路径
            additional_info: 额外信息
        """
        checkpoint = {
            'controlnet': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
        }
        
        if additional_info:
            checkpoint.update(additional_info)
        
        # 确保目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_checkpoint_memory_efficient(
        self, 
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        checkpoint_path: Path
    ) -> int:
        """
        内存高效的检查点加载
        
        Args:
            model: 要加载权重的模型
            optimizer: 优化器
            checkpoint_path: 检查点路径
            
        Returns:
            开始的epoch数
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path or not checkpoint_path.exists():
            print("No checkpoint found, starting from scratch")
            return 0
        
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        # 清理显存
        torch.cuda.empty_cache()
        
        # 加载到CPU再移动到GPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 分步骤加载，避免显存峰值
        self._load_model_state(model, checkpoint)
        self._load_optimizer_state(optimizer, checkpoint)
        
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        # 清理内存
        del checkpoint
        torch.cuda.empty_cache()
        
        print(f"Checkpoint loaded successfully. Starting from epoch {start_epoch}")
        return start_epoch
    
    def _load_model_state(self, model: nn.Module, checkpoint: Dict[str, Any]) -> None:
        """加载模型状态"""
        if 'controlnet' in checkpoint:
            controlnet_state = checkpoint['controlnet']
            model.load_state_dict(controlnet_state)
            del controlnet_state
    
    def _load_optimizer_state(
        self, 
        optimizer: torch.optim.Optimizer, 
        checkpoint: Dict[str, Any]
    ) -> None:
        """加载优化器状态"""
        if 'optimizer' in checkpoint:
            optimizer_state = checkpoint['optimizer']
            
            # 将优化器状态移动到正确的设备
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(self.device)
            
            optimizer.load_state_dict(optimizer_state)
            del optimizer_state
    
    def get_latest_checkpoint(self, checkpoint_dir: Path) -> Optional[Path]:
        """
        获取最新的检查点文件
        
        Args:
            checkpoint_dir: 检查点目录
            
        Returns:
            最新检查点的路径，如果不存在则返回None
        """
        if not checkpoint_dir.exists():
            return None
        
        checkpoint_files = list(checkpoint_dir.glob("*.pth"))
        if not checkpoint_files:
            return None
        
        # 按修改时间排序，获取最新的
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        return latest_checkpoint
    
    def cleanup_old_checkpoints(
        self, 
        checkpoint_dir: Path, 
        keep_last_n: int = 5
    ) -> None:
        """
        清理旧的检查点文件，只保留最新的N个
        
        Args:
            checkpoint_dir: 检查点目录
            keep_last_n: 保留的检查点数量
        """
        if not checkpoint_dir.exists():
            return
        
        checkpoint_files = list(checkpoint_dir.glob("controlnet_epoch*.pth"))
        
        if len(checkpoint_files) <= keep_last_n:
            return
        
        # 按epoch数排序
        checkpoint_files.sort(key=lambda p: self._extract_epoch_from_filename(p))
        
        # 删除旧的检查点
        files_to_delete = checkpoint_files[:-keep_last_n]
        for file_path in files_to_delete:
            file_path.unlink()
            print(f"Deleted old checkpoint: {file_path}")
    
    def _extract_epoch_from_filename(self, file_path: Path) -> int:
        """从文件名中提取epoch数"""
        try:
            # 假设文件名格式为 controlnet_epoch{epoch}.pth
            stem = file_path.stem  # 去掉扩展名
            epoch_str = stem.split('epoch')[-1]
            return int(epoch_str)
        except (ValueError, IndexError):
            return 0