# 记录结果/投影/
import os
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import cv2

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline

from transformers import CLIPVisionModel, CLIPImageProcessor

# from IPAdapter.ip_adapter.ip_adapter import IPAdapter
from config.train_config import AppConfig
from utils import log_util
from datasets.img_sketch_dataset import SketchControlNetDataset
import lpips


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

    # 配置日志文件
    if not args.debug:
        print('training')
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
    loss_re = nn.L1Loss()

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
    transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor()
        ])

    for epoch in range(args.num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            
            input = batch["original"].to(args.device, dtype=torch.float32)          
            sketch = batch["sketch"].to(args.device, dtype=torch.float32)
            target = batch["target"].to(args.device, dtype=torch.float32)
            mask = batch["mask"].to(args.device, dtype=torch.float32)

            # lpips_loss_fn = lpips.LPIPS(net='vgg').to(input.device)

            with torch.no_grad():
                # 预处理图像,输入到pipe中
                clip_input = clip_processor(images=input, return_tensors="pt").pixel_values.to(args.device)
                image_embeds = clip_model(clip_input).last_hidden_state  
                image_embeds = projector1(image_embeds.transpose(1, 2)).transpose(1, 2)  # [batch_size, 77, 1024]
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
                generated_images = train_pipe(
                    prompt_embeds=image_embeds,
                    pooled_prompt_embeds=image_embeds.mean(dim=1),
                    negative_prompt_embeds=torch.zeros_like(image_embeds), 
                    negative_pooled_prompt_embeds=torch.zeros_like(image_embeds.mean(dim=1)),
                    image=sketch,
                    num_inference_steps=20,
                    guidance_scale=7.5
                ).images

                # 计算生成图像和原图之间的loss
                imgs_tensor = []
                for img in generated_images:
                    img_tensor = transform(img)
                    imgs_tensor.append(img_tensor.unsqueeze(0))
                imgs_tensor = torch.cat(imgs_tensor, dim=0).to(args.device)
                valid_area = (mask < 0.1).float()
                input_valid = input * valid_area
                pred_valid = imgs_tensor * valid_area

                re_loss = loss_re(pred_valid, input_valid)
            
                total_loss = args.lam * re_loss  + noise_loss
            else: 
                total_loss = noise_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                # 测试图像
                i = random.randint(1,9)
                input_image_path = os.path.join("/home/lkh/siga/dataset/deepcad/data/cad_controlnet01/init_img", f"{i:06d}.png")
                input_sketch_path = os.path.join("/home/lkh/siga/dataset/deepcad/data/cad_controlnet01/stroke_img", f"{i:06d}.png")
                output_path = os.path.join(log_dir, "vis", f"input_{epoch}_{step}.png")
                test_img = Image.open(input_image_path).convert("RGB")
                test_sketch = Image.open(input_sketch_path).convert("RGB")
                with torch.no_grad():
                    test_sketch_ts = normlize_1(transform(test_sketch)).unsqueeze(0).to(args.device)
                    test_img_ts = transform(test_img).to((args.device))
                    test_clip_input = clip_processor(images=test_img_ts, return_tensors="pt").pixel_values.to(args.device)
                    test_img_embeds = clip_model(test_clip_input).last_hidden_state  
                    test_img_embeds = projector1(test_img_embeds.transpose(1, 2)).transpose(1, 2)  # [batch_size, 77, 1024]
                    test_img_embeds = projector2(test_img_embeds)
                    test_generated_img = train_pipe(
                        prompt_embeds=test_img_embeds,
                        pooled_prompt_embeds=test_img_embeds.mean(dim=1),
                        negative_prompt_embeds=torch.zeros_like(test_img_embeds), 
                        negative_pooled_prompt_embeds=torch.zeros_like(test_img_embeds.mean(dim=1)),
                        image=test_sketch_ts,
                        num_inference_steps=20,
                        guidance_scale=7.5
                    ).images[0]

                test_generated_img = np.array(test_generated_img)

                test_img = test_img.resize((128, 128), Image.Resampling.LANCZOS)
                test_sketch = test_sketch.resize((128, 128), Image.Resampling.LANCZOS)
                test_img = np.array(test_img)
                test_sketch = np.array(test_sketch)
                test_vis = np.concatenate([test_img, test_sketch, test_generated_img], axis=1)
                test_result = Image.fromarray(test_vis)
                test_result.save(output_path)

                # im_g.save(os.path.join(log_dir, "vis", f"input_{epoch}_{step}.png"))
                # generated_images[0].save(os.path.join(log_dir, "vis", f"pred_{epoch}_{step}.png"))

                if args.lam:
                    tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
                    tsboard_writer.add_scalar('re_loss', re_loss.item(), step)
                    log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}, re_loss: {re_loss.item():.4f}", log_file)
                else:
                    tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
                    log_util.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)

        torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))
        torch.save(train_pipe.unet.state_dict(), os.path.join(log_dir, "ckpt", f"unet_epoch{epoch}.pth"))
    torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet.pth"))
    torch.save(train_pipe.unet.state_dict(), os.path.join(log_dir, "ckpt", f"unet.pth"))
    