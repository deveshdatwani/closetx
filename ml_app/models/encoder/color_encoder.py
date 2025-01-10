import math
import colorsys
import numpy as np
from skimage.color import rgb2lab
from collections import defaultdict
from colorspacious import cspace_convert
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
import numpy


def patch_asscalar(a):
    return a.item()

setattr(numpy, "asscalar", patch_asscalar)


pink = (243,198,189) #0 
red = (209,66,71) #1 
orange = (241,89,42) #2 
beige = (245,239,221) #3 
yellow = (251,214,76) #4 
green = (1,120,72) #5 
lightblue = (200,223,236) #6 
darkblue = (26,72,113) #7 
purple = (115,71,90) #8
brown = (88,57,39) #9
grey = (174,182,189) #10
white = (0,0,0) #11
black = (255,255,255) #12
dark_beige = (217,185,155) #13


palette_rgb = {
    pink: [lightblue, darkblue, grey, white, black],
    red: [lightblue, darkblue, grey, white, black],
    orange: [green, lightblue, darkblue, white, black],
    beige: [darkblue, purple, brown, white, black],
    yellow: [green, darkblue, white, black],
    green: [orange, purple, white, black],
    lightblue: [pink, red, orange, white, black],
    darkblue: [pink, red, yellow, grey, white, black],
    purple: [orange, grey, green, white, black],
    brown: [beige, white, black],
    grey: [pink, red, darkblue, purple]
}


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


def match_color(color_to_match, colors):
    best_match = float('inf')
    best_match_key = 0
    color_to_match = rgb_to_lab(color_to_match)
    for idx, color in enumerate(colors):
        color = rgb_to_lab(color)
        adjusted_ = match_lightness(color, color_to_match)
        delta_before = delta_e_cie2000(color, color_to_match)
        delta_after = delta_e_cie2000(color, adjusted_)
        if delta_after < best_match:
            best_match = delta_after
            best_match_key = idx
    return best_match_key, best_match