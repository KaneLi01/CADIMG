### curves.py
- 定义Line, Arc, Circle及其构造方法from_dict/from_vector
- 定义以下接口（以Line为例）：
- - bbox：返回包围盒（bounding box）
  - direction()：曲线的方向（向量）
  - transform()：线性变换（平移+缩放）
  - flip()：对称翻转
  - reverse()：反转方向
  - numericalize()：数值化处理，量化为整数
  - to_vector()：转为向量表示（用于编码或储存）
  - draw()：使用 matplotlib 可视化
  - sample_points()：从曲线中采样点


### extrude.py
- 定义草图（Sketch）轮廓、坐标系和Extrude
- 进行归一化、数值化、向量化和反向还原操作。
- CoordSystem 类， 表示草图所在的平面在三维空间中的位置和朝向。  init定义了该平面的空间位置和朝向。
- Extrude 类, 对profile进行拉伸
- CADSequence 类