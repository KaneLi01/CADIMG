import sys, datetime, os, random
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils import img_utils, log_utils


from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torchvision import transforms
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline

from config.train_config import AppConfig
from dataset.dataloaders.cad_sketch_dataset import NormalSketchControlNetDataset
from models.diffusion import Diffusion_Models

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def preprocess_image(image_path, device, res):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((res, res)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


def main():
    config_json_path = '/home/lkh/siga/CADIMG/config/train_config.json'
    args = AppConfig.from_cli(config_json_path)
    if not args.debug:
        log_dir, log_file, tsboard_writer, compare_log = log_utils.setup_logdir(args.parent_log_dir, args.compare_log)  # 结果路径、tensorboard、日志文件
        AppConfig.write_config(config_obj=args, log_file=log_file, compare_log=compare_log)

    train_dataset = NormalSketchControlNetDataset(
        root_dir=args.file_path,
        mode='train',
        res=args.res
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True, # GPU加速
    )

    models = Diffusion_Models(args)

    train_pipe = StableDiffusionControlNetPipeline(
        vae=models.vae,
        text_encoder=None,
        tokenizer=None,
        unet=models.unet,
        controlnet=models.controlnet,
        scheduler=models.scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(args.device) 

    # checkpoint_path = args.resume_ckpt_path  
    # if checkpoint_path and os.path.exists(checkpoint_path):
    #     print(f"Loading ControlNet weights from {checkpoint_path}...")
    #     state_dict = torch.load(checkpoint_path, map_location=args.device)
    #     train_pipe.controlnet.load_state_dict(state_dict)    

    for param in train_pipe.unet.parameters():
        param.requires_grad = False

    loss_fn = nn.MSELoss()
    
    optimizer = optim.AdamW(
        train_pipe.controlnet.parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    ) 


    # train
    for epoch in range(args.num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            
            base = batch["base"].to(args.device, dtype=models.td)            
            sketch = batch["sketch"].to(args.device, dtype=models.td)
            target = batch["target"].to(args.device, dtype=models.td)
            # mask = batch["mask"].to(args.device, dtype=td)

            with torch.no_grad():
                # 预处理图像,输入到pipe中
                clip_input = models.clip_processor(images=base, return_tensors="pt").pixel_values.to(args.device)
                image_embeds = models.clip_model(clip_input).last_hidden_state  
                # image_embeds = projector1(image_embeds.transpose(1, 2)).transpose(1, 2)  # [1, 77, 1024]
                # image_embeds = projector2(image_embeds)
                
            # Encode target image成latents
            latents = models.vae.encode(target).latent_dist.sample() * 0.18215
            
            # 采样随机噪声，加到latent
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, models.scheduler.config.num_train_timesteps, (bsz,), device=args.device).long()
            noisy_latents = models.scheduler.add_noise(latents, noise, timesteps)

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
            optimizer.zero_grad()
            noise_loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_dir = os.path.join(args.file_path, 'val')
                test_img_dir = os.path.join(test_dir, 'base_img')
                test_sketch_dir = os.path.join(test_dir, 'sketch_img')
                all_files = [f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f))]
                selected_files = random.sample(all_files, 4)
                output_list = []
                for fname in selected_files:
                    img_path = os.path.join(test_img_dir, fname)
                    sketch_path = os.path.join(test_sketch_dir, fname)
                    test_image = preprocess_image(img_path, args.device, args.res)
                    test_sketch_image = transforms.Normalize([0.5], [0.5])(preprocess_image(sketch_path, args.device, args.res))
                    with torch.no_grad():
                        # 处理输入图像
                        test_clip_input = models.clip_processor(images=test_image, return_tensors="pt").pixel_values.to(args.device)
                        test_image_embeds = models.clip_model(test_clip_input).last_hidden_state
                        prompt_embeds = torch.cat([test_image_embeds, test_image_embeds], dim=0)
                        pooled_prompt_embeds = test_image_embeds.mean(dim=1)
                        output = train_pipe(
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            negative_prompt_embeds=torch.zeros_like(prompt_embeds),
                                                                                                                    
                            image=test_sketch_image,
                            num_inference_steps=20,
                            guidance_scale=7.5,
                        ).images[0]
                    output_list.append(output)

                img_utils.merge_imgs(output_list, os.path.join(log_dir, "vis", f"{epoch}_{step}.png"))
                global_step = epoch * len(train_dataloader) + step
                tsboard_writer.add_scalar('noise_loss', noise_loss.item(), global_step)
                log_utils.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)
        if epoch % 5 == 0:
            torch.save({
                'controlnet': train_pipe.controlnet.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
            }, os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))
            
    torch.save({
        'controlnet': train_pipe.controlnet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': args.num_epochs,
    }, os.path.join(log_dir, "ckpt", "controlnet.pth"))

            

if __name__ == "__main__":
    main()


    # # 其他
    # # 将img编码进行投影，以符合stable diffusion的输入
    # projector1 = nn.Linear(257, 77).to(args.device)
    # projector2 = nn.Linear(1024, 768).to(args.device)
    # projector1.load_state_dict(projector_ckpt['projector_257to77'])
    # projector2.load_state_dict(projector_ckpt['projector_1024to768'])
    # # 用于重建loss，和target对齐
    # normlize_1 = transforms.Normalize([0.5], [0.5])
    raise Exception('over')


            


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







        if args.lam:
            tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
            tsboard_writer.add_scalar('re_loss', re_loss.item(), step)
            log_utils.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}, re_loss: {re_loss.item():.4f}", log_file)
        else:
            tsboard_writer.add_scalar('noise_loss', noise_loss.item(), step)
            log_utils.log_string(f"Epoch {epoch}, Step {step}, noise_loss: {noise_loss.item():.4f}", log_file)

        torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth"))

    torch.save(train_pipe.controlnet.state_dict(), os.path.join(log_dir, "ckpt", f"controlnet.pth"))

    