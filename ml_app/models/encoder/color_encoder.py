import numpy 
def patch_asscalar(a):
    return a.item()

import numpy as np
from colorspacious import cspace_convert
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from .color_config import *
setattr(numpy, "asscalar", patch_asscalar)

palette_rbg_list = list(palette_rgb.keys())

def rgb_to_ycc(r, g, b):
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    return y, cb, cr


def hyab_distance(color1, color2):
    cam1 = cspace_convert(color1, "sRGB1", "CAM02-UCS")
    cam2 = cspace_convert(color2, "sRGB1", "CAM02-UCS")
    def to_hyab(cam):
        H = cam[0]  
        y = cam[1]  
        ab = cam[2] 
        return np.array([H * 1.2, y, ab])  
    hyab1 = to_hyab(cam1)
    hyab2 = to_hyab(cam2)    
    return np.linalg.norm(hyab1 - hyab2)


def to_ycc(color): 
    return rgb_to_ycc(*[x/255.0 for x in color])


def color_dist(c1, c2):
    return sum((a-b)**2 for a, b in zip(to_ycc(c1), to_ycc(c2)))


def match_lightness(light_color, dark_color):
    adjusted_dark_brown = LabColor(
        lab_l=light_color.lab_l,  
        lab_a=dark_color.lab_a,
        lab_b=dark_color.lab_b
    )
    return adjusted_dark_brown


def rgb_to_lab(color):
    r,g,b = color[0], color[1], color[2]
    rgb_color = sRGBColor(r / 255.0, g / 255.0, b / 255.0)
    lab_color = convert_color(rgb_color, LabColor)
    return lab_color


def match_color(color_to_match=None, colors=palette_rgb):
    best_match = float('inf')
    best_match_key = 0
    color_to_match = rgb_to_lab(color_to_match)
    for idx, color in enumerate(colors):
        color = rgb_to_lab(color)
        adjusted_ = match_lightness(color, color_to_match)
        delta_after = delta_e_cie2000(color, adjusted_)
        if delta_after < best_match:
            best_match = delta_after
            best_match_key = idx
    return best_match_key


def apprel_seg(model, img, type):
    masks = model(img)