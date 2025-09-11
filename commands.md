## linux
### 目录、文件操作
移动目录A到B下（包含所有子目录和文件）：mv A B/ 

显示路径下所有文件夹： ls -a

统计路径下所有文件和文件夹：  ls -a | wc -l

解压文件到指定目录： unzip 文件名.zip -d 目标目录

解压tar文件：tar -xvf 文件名.tar -C 指定目录

解压tar.gz文件：tar -xzvf 文件名.tar.gz

查看目录下的文件和文件夹大小，按照从小到大排序：du -h --max-depth=1 /var | sort -h

删除目录下某种文件：rm -v /path/to/directory/*.png

删除目录下所有文件：rm -rf

查看目录下所有文件大小： du -sh .

权限：给shell添加读写权限： chmod +x xxx.sh

递归查看所有空子目录： find . -type d -empty (要删除再添加参数-delete)

递归统计目录下所有子文件数量： find . -type f | wc -l


### bash
read a 将命令行变量写入a，可以用于暂停

## 传输数据
从服务器下载数据：rsync -avz -e 'ssh -p 3614' lkh@59.77.13.211:/home/lkh/siga/dataset/my_dataset/cad_rgb_imgs/cad_controlnet_cube_dark/train /home/kane/code/datasets/cad_controlnet_cube_dark/

### 密钥

### 进程
根据 PID查看 用户：ps -o user= -p <PID>

### 后台运行
后台运行：nohup python your_script.py > output.log 2>&1 &
nohup python /home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/postprocess/tasks/c1_t3_postpro.py > ./src/output0902.log 2>&1 &
终止：ps aux | grep "python your_script.py" ；kill -9 PID
ps aux | grep "bash /home/lkh/siga/CADIMG/experiments/train.sh" ；kill -9 PID

### 离线渲染
xvfb-run -a python your_script.py
cd /home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/render
xvfb-run -a python c1_pipeline.py -r 00800000 00999999 --config /home/lkh/siga/CADIMG/dataset/data_pipeline/ABC/render/configs/render_normal_1.yaml
00581401 00791778
---

## python：
创建文件夹：os.makedirs("new_folder", exist_ok=True)

---

## github：

删除remote已经提交的内容：git rm --cached 文件, 然后再git push

查看本地提交历史：git log --oneline (按q退出查看)

---

## huggingface
export HF_ENDPOINT=https://hf-mirror.com (已经写入到.bashrc文件中)

huggingface-cli download stable-diffusion-v1-5/stable-diffusion-v1-5 --local-dir /home/kane/code/models_weights/stable-diffusion-v1-5 --force-download

---

## blenderproc

将h5文件渲染为图片： blenderproc vis hdf5 "/home/lkh/siga/output/temp/bl/0.hdf5" --keys "normals" --save "/home/lkh/siga/output/temp"