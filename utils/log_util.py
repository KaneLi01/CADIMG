import os
import datetime
import torch
from torch.utils.tensorboard import SummaryWriter



def make_new_sorted_file(path, file_name):
    # 根据父目录下的文件夹数量命名子文件夹
    if not os.path.exists(path):
        raise ValueError(f"{path} is not a vaild path")
    exist_num = sum(os.path.isdir(os.path.join(path, item)) for item in os.listdir(path))
    new_file_path = os.path.join(path, file_name + f'{exist_num:02d}') 
    os.mkdir(new_file_path)
    print(f"created the {file_name}")

    return new_file_path


def log_string(out_str, log_file):
    # 将传入str输入到log文件中
    log_file.write(out_str+'\n')
    log_file.flush()
    print(out_str)


def log_losses(writer, batch_idx, num_batches, loss_dict):
    # helper function to log losses to tensorboardx writer
    iteration = batch_idx + 1
    for loss in loss_dict.keys():
        writer.add_scalar(loss, loss_dict[loss].item(), iteration)
    return iteration


def setup_logdir(parent_log_dir, compare_log):

    # 创建日志路径、tensorboard、输出的文件名称
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M")
    log_dir = os.path.join(parent_log_dir, timestamp)
    tsboard_writer = SummaryWriter(log_dir)
    log_filename = os.path.join(log_dir, 'output.log')
    log_file = open(log_filename, 'w')
    os.mkdir(os.path.join(log_dir, "vis"))
    os.mkdir(os.path.join(log_dir, "ckpt"))
    compare_log_file = open(compare_log, 'a')
    compare_log_file.write(f"\n─────────────────────")
    compare_log_file.write(f"\n{timestamp} |\t")

    return log_dir, log_file, tsboard_writer, compare_log_file
