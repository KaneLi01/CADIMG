import hakowan as hkw
import numpy as np


config = hkw.config()  
# config.sensor = hkw.setup.sensor.Orthographic()
config.sensor.location = [-1,2,-3]

print(config.sensor)

base = hkw.layer("/home/lkh/siga/output/temp/wireframe.stl")  

hkw.render(base, config=config, filename="image_wire.png")  # Render!