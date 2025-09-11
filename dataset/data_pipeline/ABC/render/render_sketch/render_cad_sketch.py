from OCC.Display.OCCViewer import Viewer3d
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB, Quantity_NOC_BLACK
from OCC.Core.Prs3d import Prs3d_Drawer, Prs3d_LineAspect
from OCC.Core.Aspect import Aspect_TOL_DASH
from OCC.Core.AIS import AIS_Shape
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve

import utils.cadlib.Brep_utils as Brep_utils

import gc

class SketchOffscreenRenderer:
    """
    离线3D渲染器类，用于渲染形状、线框或二者合并
    """
    
    def __init__(self):
        self.renderer = None
        self._setup_renderer(light=True)
        self.drawer = self._create_dash_line_drawer()

    def _setup_renderer(self, light=True):
        """设置基础渲染器"""

        self.renderer = Viewer3d()
        if light:  
            self.renderer.Create()  
        else:
            self.renderer.Create(phong_shading=False, create_default_lights=False)
        self.renderer.SetModeShaded()
        self.renderer.SetSize(512, 512)
        

        self.renderer.SetOrthographicProjection()

        # 渲染背景色
        color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
        self.renderer.View.SetBgGradientColors(color, color)

    def _set_camera(self, cam_pos=None, see_at=None):
        """
        设置摄像机位置和目标点
        
        Args:
            cam_pos (tuple): 摄像机位置 (x, y, z)
            see_at (tuple): 摄像机目标点 (x, y, z)
        """
        if cam_pos is not None:
            self.renderer.View.SetEye(cam_pos[0], cam_pos[1], cam_pos[2])
        if see_at is not None:
            self.renderer.View.SetAt(see_at[0], see_at[1], see_at[2])

    def _create_dash_line_drawer(self):
        """创建虚线样式的绘制器"""
        drawer = Prs3d_Drawer()
        line_aspect = Prs3d_LineAspect(
            Quantity_Color(Quantity_NOC_BLACK),
            Aspect_TOL_DASH,
            2.0
        )
        drawer.SetWireAspect(line_aspect)
        return drawer
    
    def render_dash_wires(self, output_path, shape, scale, cam_pos, see_at):
        """
        渲染虚线线框
        """

        if not shape:
            raise Exception('input at least one item!')
        # self._setup_renderer(light=True, scale=scale)
        self._set_camera(cam_pos=cam_pos, see_at=see_at)  
        self.renderer.View.SetScale(scale)     
        
        # 设置白色背景
        bg_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
        self.renderer.View.SetBgGradientColors(bg_color, bg_color)
        
        # 显示所有边
        edge_list = Brep_utils.get_edges(shape)
        for edge in edge_list:
            edge_adaptor = BRepAdaptor_Curve(edge)
            edge_type = edge_adaptor.GetType()            
            if edge_type != 6:
                ais_shape = AIS_Shape(edge)
                ais_shape.SetAttributes(self.drawer)
                self.renderer.Context.Display(ais_shape, False)
        # c = self.renderer.GetCamera()
        # e = c.Eye()
        # ce = c.Center()
        self.renderer.View.Dump(output_path)
        self.renderer.Context.EraseAll(True)


    def render_shape_with_wires(self, output_path, shape, scale, cam_pos, see_at):
        """
        渲染带白色面的形状和线框
        """

        if shape is None:
            raise Exception('input at least one item!')

        # self._setup_renderer(light=True, scale=scale)
        self._set_camera(cam_pos=cam_pos, see_at=see_at)  
        self.renderer.View.SetScale(scale)     
        
        # 渲染形状（白色面）
        if shape is not None:
            white_color = Quantity_Color(1.0, 1.0, 1.0, Quantity_TOC_RGB)
            self.renderer.DisplayShape(shape, update=False, color=white_color)
        
        # 渲染线框（黑色线）
        edge_list = Brep_utils.get_edges(shape)
        if edge_list is not None:
            for wire in edge_list:
                edge_adaptor = BRepAdaptor_Curve(wire)
                edge_type = edge_adaptor.GetType()            
                if edge_type != 6:                
                    ais_wire = AIS_Shape(wire)
                    ais_wire.SetWidth(3.0)
                    ais_wire.SetColor(Quantity_Color(Quantity_NOC_BLACK))
                    self.renderer.Context.Display(ais_wire, False)
   
        self.renderer.View.Dump(output_path)
        self.renderer.Context.EraseAll(True)


    def render_shape_black(self, output_path, shape, scale, cam_pos, see_at):
        """
        渲染纯黑的形状
        """
        
        if shape is None:
            raise Exception('input at least one item!')

        # self._setup_renderer(light=True, scale=scale)
        self._set_camera(cam_pos=cam_pos, see_at=see_at)   
        self.renderer.View.SetScale(scale) 
        
        # 渲染纯黑面
        if shape is not None:
            black_color = Quantity_Color(0.0, 0.0, 0.0, Quantity_TOC_RGB)
            self.renderer.DisplayShape(shape, update=False, color=black_color)

        self.renderer.View.Dump(output_path)
        self.renderer.Context.EraseAll(True)


        #self._clear_renderer()


