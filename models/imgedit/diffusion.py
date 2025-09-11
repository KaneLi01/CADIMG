import torch
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionModelWithProjection
from diffusers import StableDiffusionPipeline, ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline


class Diffusion_Models:
    def __init__(self, args):

        self.td = torch.float32
        if hasattr(args, 'torch_dtype'):
            if args.torch_dtype == 'torch16':
                self.td = torch.float16            

        self.clip_model = CLIPVisionModel.from_pretrained(args.img_encoder_path).to(args.device)
        self.clip_processor = CLIPImageProcessor.from_pretrained(args.img_encoder_path)

        self.pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_path,
            torch_dtype=self.td,  
            safety_checker=None,        
            requires_safety_checker=False
            )

        self.vae = self.pipe.vae.to(args.device)
        self.unet = self.pipe.unet.to(args.device)
        self.scheduler = self.pipe.scheduler
        # self.text_encoder = self.pipe.text_encoder.to(args.device)
        # self.tokenizer = self.pipe.tokenizer

        self.controlnet = ControlNetModel.from_pretrained(
            args.controlnet_path,
            torch_dtype=self.td
        )    

        self.projector_ckpt = torch.load(args.projector_path, map_location=args.device)






