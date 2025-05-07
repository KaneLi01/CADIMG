class ARGS:
    def __init__(self):
        self.file_path = "/home/lkh/siga/dataset/deepcad/data/cad_controlnet02"  # 数据集路径
        self.parent_log_dir = "/home/lkh/siga/CADIMG/log"  # 结果保存总目录
        self.controlnet_path = "/home/lkh/siga/ckpt/controlnet_scribble"
        self.sd_path = "/home/lkh/siga/ckpt/sd15" 
        self.img_encoder_path = "/home/lkh/siga/ckpt/clip-vit-large-patch14"
        self.projector_path = "/home/lkh/siga/ckpt/projector_weights.pth"

        self.device = "cuda:0"
        self.num_epochs = 10
        self.batch_size = 4
        self.lam = 0.2

        # controlnet model
        # float16/32
        