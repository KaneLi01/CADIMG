# 记录结果/投影/
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os
import torch
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from torch import nn
import torch.optim as optim
from torchvision.utils import save_image
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision import transforms
from transformers import CLIPVisionModel, CLIPImageProcessor
# from IPAdapter.ip_adapter.ip_adapter import IPAdapter
from train_config import AppConfig

from utils import log_util
from datasets.controlnet_train import SketchControlNetDataset

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def load_models(args):

    clip_model = CLIPVisionModel.from_pretrained(args.img_encoder_path).to(args.device)
    clip_processor = CLIPImageProcessor.from_pretrained(args.img_encoder_path)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float32,  
        safety_checker=None,        
        requires_safety_checker=False
        )
    vae = pipe.vae.to(args.device)
    unet = pipe.unet.to(args.device)
    scheduler = pipe.scheduler

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_path,
        torch_dtype=torch.float32
    )    

    projector_ckpt = torch.load(args.projector_path, map_location=args.device)

    return clip_model, clip_processor, pipe.to(args.device), vae, unet, scheduler, controlnet.to(args.device), projector_ckpt



if __name__ == "__main__":
    # 读取参数
    args = AppConfig.from_cli()
    print(type(args.lam))

    # 配置日志文件
    log_dir, log_file, tsboard_writer, compare_log = log_util.setup_logdir(args.parent_log_dir, args.compare_log)  # 结果路径、tensorboard、日志文件
    AppConfig.write_config(config_obj=args, log_file=log_file, compare_log=compare_log)

    # 数据集
    train_dataset = SketchControlNetDataset(
        root_dir=args.file_path  
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, # GPU加速
    )

    # 读取各个模块
    clip_model, clip_processor, pipe, vae, unet, scheduler, controlnet, projector_ckpt = \
        load_models(args)

    # 组装pipeline
    train_pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        unet=unet,
        controlnet=controlnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None
    )
    train_pipe = train_pipe.to(args.device)   

    # 定义损失和优化器
    loss_fn = nn.MSELoss()

    optimizer = optim.AdamW(
        train_pipe.controlnet.parameters(),
        lr=2e-5, weight_decay=1e-2
    ) 

    # 其他
    # 将img编码进行投影，以符合stable diffusion的输入
    projector1 = nn.Linear(257, 77).to(args.device)
    projector2 = nn.Linear(1024, 768).to(args.device)
    projector1.load_state_dict(projector_ckpt['projector_257to77'])
    projector2.load_state_dict(projector_ckpt['projector_1024to768'])
    # 用于重建loss，和target对齐
    normlize_1 = transforms.Normalize([0.5], [0.5])


    for epoch in range(args.num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            
            input = batch["original"].to(args.device, dtype=torch.float32)            
            sketch = batch["sketch"].to(args.device, dtype=torch.float32)
            target = batch["target"].to(args.device, dtype=torch.float32)
            mask = batch["mask"].to(args.device, dtype=torch.float32)

            with torch.no_grad():
                # 预处理图像,输入到pipe中
                clip_input = clip_processor(images=input, return_tensors="pt").pixel_values.to(args.device)
                image_embeds = clip_model(clip_input).last_hidden_state  
                image_embeds = projector1(image_embeds.transpose(1, 2)).transpose(1, 2)  # [1, 77, 1024]
                image_embeds = projector2(image_embeds)
                
            # Encode target image成latents
            latents = vae.encode(target).latent_dist.sample() * 0.18215
            
            # 采样随机噪声，加到latent
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # 控制条件
            controlnet_conditioning_image = sketch  # sketch作为control hint

            # controlnet 向前传播
            down_block_res_samples, mid_block_res_sample = train_pipe.controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=image_embeds,  # 提示嵌入
                controlnet_cond=sketch,  # 控制， 该模型 提示使用3通道图像
                return_dict=False,
            )

            noise_pred = train_pipe.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=image_embeds,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # 计算loss
            noise_loss = loss_fn(noise_pred, noise)



            # 非mask区域的重建loss
            if args.lam:
                pred_latents = noisy_latents - noise_pred
                pred_image = train_pipe.vae.decode(pred_latents / 0.18215).sample  #  预测图像

                valid_area = (mask < 0.1).float() 
                original_valid = normlize_1(input) * valid_area
                pred_valid = pred_image * valid_area

                re_loss = loss_fn(pred_valid, original_valid)
            
                total_loss = args.lam * re_loss  + noise_loss
            else: 
                total_loss = noise_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                # if step % 200 == 0:
                #     save_image(pred_image, os.path.join(log_dir, "vis", f"{epoch}_{step}.png"))

                if args.lam:
                    tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
                    tsboard_writer.add_scalar('re_loss', re_loss.item(), step)
                    log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}, re_loss: {re_loss.item():.4f}", log_file)
                else:
                    tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
                    log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)

        # torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))
        # torch.save(train_pipe.unet.state_dict(), os.path.join(log_dir, "ckpt", f"unet_epoch{epoch}.pth"))
    torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet.pth"))
    torch.save(train_pipe.unet.state_dict(), os.path.join(log_dir, "ckpt", f"unet.pth"))
    