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
from models.imgedit.diffusion import Diffusion_Models
from engine.imgedit.checkpoint_manager import CheckpointManager
from engine.imgedit.data_processor import DataProcessor
from models.imgedit.model_setup import ModelSetup

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main():
    config_json_path = '/home/lkh/siga/CADIMG/config/train_config.json'
    args = AppConfig.from_cli(config_json_path)
    if not args.debug:
        log_dir, log_file, tsboard_writer, compare_log = log_utils.setup_logdir(args.parent_log_dir, args.compare_log)  # 结果路径、tensorboard、日志文件
        AppConfig.write_config(config_obj=args, log_file=log_file, compare_log=compare_log)

    cm = CheckpointManager(device=args.device)
    dp = DataProcessor(device=args.device, resolution=args.res)
    ms = ModelSetup(args)
    transf = transforms.ToTensor()

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

    ms.initialize_models()
    train_pipe = ms.create_training_pipeline()
    loss_fn, loss_fn_x = ms.setup_loss_function()
    optimizer = ms.setup_optimizer()

    checkpoint_path = args.resume_ckpt_path
    if checkpoint_path and os.path.exists(checkpoint_path):
        cm.load_checkpoint_memory_efficient(
            model=train_pipe.controlnet,
            optimizer=optimizer,
            checkpoint_path=args.resume_ckpt_path          
        )

    # train
    for epoch in range(args.num_epochs):
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch}")
        for step, batch in loop:
            train_data = dp.process_training_batch(batch=batch, 
                                                   models=ms.models, 
                                                   target_dtype=ms.models.td)
            mask = batch["mask"].to(args.device, dtype=ms.models.td)
            
            noisy_latents = ms.models.scheduler.add_noise(train_data['latents'], train_data['noise'], train_data['timesteps'])

            # controlnet 向前传播
            down_block_res_samples, mid_block_res_sample = train_pipe.controlnet(
                noisy_latents,
                train_data['timesteps'],
                encoder_hidden_states=train_data['image_embeds'],  # 提示嵌入
                controlnet_cond=train_data['sketch'],  # 控制， 该模型 提示使用3通道图像
                return_dict=False,
            )

            noise_pred = train_pipe.unet(
                noisy_latents,
                train_data['timesteps'],
                encoder_hidden_states=train_data['image_embeds'],
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            pred_latents = noisy_latents - noise_pred
            pre_img = ms.models.vae.decode(pred_latents / ms.models.vae.config.scaling_factor).sample
            target = batch["target"].to(args.device, dtype=ms.models.td)

            # 计算loss
            loss1 = loss_fn(noise_pred, train_data['noise'])
            loss2 = loss_fn_x(pre_img*mask, target*mask)
            
            #loss2 = loss_fn_x(pre_img, target)
            noise_loss = loss1+loss2
            
            optimizer.zero_grad()
            noise_loss.backward()
            torch.nn.utils.clip_grad_norm_(train_pipe.controlnet.parameters(), max_norm=1.0)
            optimizer.step()

            if step % 50 == 0:
                test_dir = os.path.join(args.file_path, 'val')
                test_img_dir = os.path.join(test_dir, 'base_img')
                test_sketch_dir = os.path.join(test_dir, 'sketch_img')
                all_files = [f for f in os.listdir(test_img_dir) if os.path.isfile(os.path.join(test_img_dir, f))]
                selected_files = random.sample(all_files, 4)
                output_list = []
                input_list = []
                for fname in selected_files:
                    img_path = os.path.join(test_img_dir, fname)
                    sketch_path = os.path.join(test_sketch_dir, fname)
                    input_vis = dp.create_input_visualization(img_path=img_path, sketch_path=sketch_path)
                    input_list.append(input_vis)
                    
                    test_data = dp.process_test_images(img_path=img_path, sketch_path=sketch_path, models=ms.models)

                    output = train_pipe(
                        prompt_embeds=test_data['prompt_embeds'],
                        pooled_prompt_embeds=test_data['pooled_embeds'],
                        negative_prompt_embeds=torch.zeros_like(test_data['prompt_embeds']),
                                                                                                                
                        image=test_data['test_sketch'],
                        num_inference_steps=20,
                        guidance_scale=7.5,
                    ).images[0]
                    output_list.append(output)

                img_utils.merge_imgs(input_list+output_list, os.path.join(log_dir, "vis", f"{epoch}_{step}.png"), mode='grid', grid_size=(2, 4))
                global_step = epoch * len(train_dataloader) + step
                # tsboard_writer.add_scalar('loss1', loss1.item(), global_step)
                tsboard_writer.add_scalar('loss', noise_loss.item(), global_step)
                log_utils.log_string(f"Epoch {epoch}, Step {step}, loss1: {loss1.item():.4f}, loss2: {loss2.item():.4f}", log_file)
                #log_utils.log_string(f"Epoch {epoch}, Step {step}, loss: {noise_loss.item():.4f}", log_file)

        if epoch % 1 == 0:
            cm.save_checkpoint(
                model=train_pipe.controlnet,
                optimizer=optimizer,
                epoch=epoch,
                save_path=os.path.join(log_dir, "ckpt", f"controlnet_epoch{epoch}.pth")
            )
    cm.save_checkpoint(
        model=train_pipe.controlnet,
        optimizer=optimizer,
        epoch=epoch,
        save_path=os.path.join(log_dir, "ckpt", "controlnet.pth")
    )            
            

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

    