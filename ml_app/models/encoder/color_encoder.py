import math
import numpy 
def patch_asscalar(a):
    return a.item()

import numpy as np
from colorspacious import cspace_convert
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from .palette import *
setattr(numpy, "asscalar", patch_asscalar)


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


def get_palette_color(color_to_match=None, colors=palette_rgb, get_idx=False):
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
            best_match_color = palette_rgb[idx]
    if get_idx:
        return best_match_key
    return palette_names[best_match_color]


def apprel_seg(model, img, type):
    masks = model(img)


def match_apparel_color(r1, g1, b1, r2, g2, b2):
    color_1 = get_palette_color((r1,g1,b1))
    color_2 = get_palette_color((r2,g2,b2))  
    if palette_match[palette_names[color_1]] in palette_match[palette_names[color_2]]: 
        return 100
    else: return 0


def rgb_to_xyz(rgb):
    r, g, b = rgb
    r /= 255.0
    g /= 255.0
    b /= 255.0
    r = r / 12.92 if r <= 0.04045 else ((r + 0.055) / 1.055) ** 2.4
    g = g / 12.92 if g <= 0.04045 else ((g + 0.055) / 1.055) ** 2.4
    b = b / 12.92 if b <= 0.04045 else ((b + 0.055) / 1.055) ** 2.4
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    x /= 0.95047
    z /= 1.08883
    return (x, y, z)


def xyz_to_lab(xyz):
    x, y, z = xyz
    ref_x, ref_y, ref_z = 0.95047, 1.00000, 1.08883
    x = x / ref_x if x > 0.008856 else (x * 903.3 + 16) / 116
    y = y / ref_y if y > 0.008856 else (y * 903.3 + 16) / 116
    z = z / ref_z if z > 0.008856 else (z * 903.3 + 16) / 116
    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)    
    return (l, a, b)


def cie_de2000(c1, c2):
    lab1 = xyz_to_lab(rgb_to_xyz(c1))
    lab2 = xyz_to_lab(rgb_to_xyz(c2))
    L1, A1, B1 = lab1
    L2, A2, B2 = lab2
    C1 = math.sqrt(A1**2 + B1**2)
    C2 = math.sqrt(A2**2 + B2**2)
    delta_L = L2 - L1
    delta_C = C2 - C1
    delta_A = A2 - A1
    delta_B = B2 - B1
    delta_H = delta_A**2 + delta_B**2 - delta_C**2
    if delta_H < 0:
        delta_H = math.sqrt(delta_A**2 + delta_B**2)
    else:
        delta_H = math.sqrt(delta_H)
    k_L = 1.0
    k_C = 1.0
    k_H = 1.0
    delta_E = math.sqrt(
        (delta_L / k_L)**2 + (delta_C / k_C)**2 + (delta_H / k_H)**2
    )
    return delta_E

def match_colors(colors=palette_rgb, match_color=None):
    min_dist = float('inf')
    best_match = None
    for color in colors:
        dist = cie_de2000(color, match_color)
        if dist < min_dist:
            min_dist = dist
            best_match = color
    return best_match