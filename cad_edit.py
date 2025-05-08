import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline
from transformers import CLIPVisionModel, CLIPImageProcessor
from cad_edit_config import AppConfig
import os
from torch import nn

# 加载模型函数
def load_inference_models(args):
    # 加载CLIP模型用于图像编码
    clip_model = CLIPVisionModel.from_pretrained(args.img_encoder_path).to(args.device)
    clip_processor = CLIPImageProcessor.from_pretrained(args.img_encoder_path)
    
    # 加载基础Stable Diffusion模型
    pipe = StableDiffusionPipeline.from_pretrained(
        args.sd_path,
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # 加载训练好的ControlNet
    controlnet = ControlNetModel.from_pretrained(args.controlnet_path)
    trained_cn_path = os.path.join(args.parent_cn_path, args.index, "ckpt/controlnet.pth")  # 如果不是则需要修改
    controlnet.load_state_dict(torch.load(trained_cn_path))  
    
    # 加载投影器
    projector_ckpt = torch.load(args.projector_path, map_location=args.device)
    projector1 = nn.Linear(257, 77).to(args.device)
    projector2 = nn.Linear(1024, 768).to(args.device)
    projector1.load_state_dict(projector_ckpt['projector_257to77'])
    projector2.load_state_dict(projector_ckpt['projector_1024to768'])
    
    # 组装推理pipeline

    # prompt = [" "] 
    # input_ids = train_pipe.tokenizer(
    #         prompt,
    #         padding="max_length",
    #         max_length=train_pipe.tokenizer.model_max_length,
    #         truncation=True,
    #         return_tensors="pt",
    #     ).input_ids.to(args.device)

    inference_pipe = StableDiffusionControlNetPipeline(
        vae=pipe.vae,
        text_encoder=None,
        tokenizer=None,
        unet=pipe.unet,
        controlnet=controlnet,
        scheduler=pipe.scheduler,
        safety_checker=None,
        feature_extractor=None
    ).to(args.device)
    
    return clip_model, clip_processor, inference_pipe, projector1, projector2

# 图像预处理
def preprocess_image(image_path, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)

# 推理函数
def infer(input_image_path, sketch_image_path, output_path, models, num_inference_steps=20, guidance_scale=7.5):
    clip_model, clip_processor, pipe, projector1, projector2 = models
    
    # 预处理输入图像
    input_image = preprocess_image(input_image_path, args.device)
    sketch_image = transforms.Normalize([0.5], [0.5])(preprocess_image(sketch_image_path, args.device))
    
    # 生成图像嵌入
    with torch.no_grad():
        # 处理输入图像
        clip_input = clip_processor(images=input_image, return_tensors="pt").pixel_values.to(args.device)
        image_embeds = clip_model(clip_input).last_hidden_state
        image_embeds = projector1(image_embeds.transpose(1, 2)).transpose(1, 2)
        image_embeds = projector2(image_embeds)

        prompt_embeds = torch.cat([image_embeds, image_embeds], dim=0)
        pooled_prompt_embeds = image_embeds.mean(dim=1)

        # 生成图像
        output = pipe(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_prompt_embeds=torch.zeros_like(prompt_embeds),  # 简单使用零作为负提示
            negative_pooled_prompt_embeds=torch.zeros_like(pooled_prompt_embeds),
            image=sketch_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]
        
    # 保存结果
    output.save(output_path)
    return output

if __name__ == "__main__":
    args = AppConfig.from_cli()

    # 加载模型
    models = load_inference_models(args)
    
    # 推理
    for i in range(args.img_index[0],args.img_index[1]):
        input_image_path = os.path.join(args.test_img_dir, f"{i:06d}.png")
        sketch_image_path = os.path.join(args.test_sketch_dir, f"{i:06d}.png")          
        
        output_path = os.path.join(args.output_dir, "lam", f"{args.index}_{i}.png")  # 输出图像路径
        
        result = infer(input_image_path, sketch_image_path, output_path, models)
        
        print(f"Inference completed. Result saved to {output_path}")