# DeepCAD 上的数据集制作和训练

- DeepCAD数据集 共有215093个json文件，每个json文件代表由extrude创建的shape，其中记录构造shape的command和parameter，以line/circle/arc为基础创建loop->profile->extrusion
- 所有形状的最小**包围盒**都**平行坐标轴**，且范围在[-0.75,0.75]^3之间


## 数据筛选
按照以下要求对json文件中记录的数据进行筛选，然后记录对应文件名。

1.  extrusion操作对象数量为2，即**两个body**
2.  两个body之间的操作类型为**并集**
3.  两个body之间**相切**。通过设置**最小距离**小于1e-5和**相交体积**小于1e-5筛选（pythonocc库API）
4.  **薄片和棍状**的body删去。获取shape的包围盒，假如 最短边*15 < 最长边，则删去（pythonocc库API）
5.  获取最终的shape，遍历所有的面，当**面数大于30**时删去（pythonocc库API）


## 渲染
- 每个shape渲染**base/sketch/target**图片用于训练；每个类型的图片渲染**6个不同相机位姿**图
    - 每个shape包含两个body，第一个body作为base形状，通过**blenderproc**库渲染其normal图像；第二个body用于模拟sketch，通过**pythonocc**库渲染其**wireframe**；拟合的target图像是渲染整个shape的normal图像。不同库渲染时，需要进行**相机位姿的坐标变换**。
    - 获取两个body的包围盒，比较**包围盒的相对位置**，并确定相机观测**视角为front/back/up/down/left/right**其中之一，以确保表示sketch的第二个body在该相机视角下**不会被遮挡**。具体方式是
        - 获取两个body包围盒的**最小点p_min和最大点p_max**，如果body2的p_min的z值，恒大于body1的p_max的z值（设置一个误差容忍范围1e-5），那么说明body2一定在body1的上方，设置该相机位姿类型为up。其他方位类似
        - 假如相机位姿类型为up，那么在shape上方设置满足**wonder3d**的6个相机坐标`[0.0,-2*sqrt(2),2.0], [2.0,-2.0,2.0], [2*sqrt(2),0,2.0], [2.0,2.0,2.0], [0.0,2*sqrt(2),2.0], [-2*sqrt(2),0.0,2.0]`，即坐标点都在z=2的平面上的，半径为$2\sqrt2$，圆心为(0,0,2)的圆上。其他视角方向同理。
- 对于profile是圆/圆弧的body，由于用pythonocc渲染时只会绘制**一条侧边**的线用于表示extrude操作（通常是圆绘制的起点），不符合用户绘制圆柱sketch的逻辑。因此对这类对象进行特殊处理
    - 筛选出包含**circle的loop**，并记录这些由这些loop extrude的body，然后遍历其中所有edge，并**删去线段**，然后渲染sketch图。这样得到的圆柱sketch仅有表示**两个底面的圆形**。arc同理。
    - 圆柱的侧面边界，通过渲染相同body的normal图，然后**提取边界**后**叠加**在上面绘制的sketch图上


## 训练设置

#### 模型
**pipe**: StableDiffusionControlNetPipeline

- **stable diffusion**: stable-diffusion-v1-5/table-diffusion-v1-5,
- **controlnet**: lllyasviel/sd-controlnet-canny,
- **img encoder**: openai/clip-vit-base-patch32,



#### 训练超参数
-   res: 256,
-   num_epochs: 10,
-   batch_size: 4,
-   torch_dtype: float32,
-   optim: AdamW,
-   lr: 1e-5,
-   weight_decay: 1e-2,
-   loss: MSE

#### 实验环境

single RTX 3090，24G显存

根据上面的参数设置，大约23G，一个epoch需要40min左右





