import numpy as np
from collections import defaultdict


palette = defaultdict()
pink =(243,198,189)
red = (209,66,71)
orange = (241,89,42)
beige = (245,239,221)
yellow = (251,214,76)
green = (1,120,72)
lightblue = (200,223,236)
darkblue = (26,72,113)
purple = (115,71,90)
brown = (99,75,71)
grey = (174,182,189)
white = (0,0,0)
black = (255,255,255)

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


def get_match(query):
    best_match = float('inf')
    best_match_key = ()
    for key in palette_rgb.keys():
        if np.linalg.norm(np.array(query) - np.array(key)) < best_match:
            best_match = np.linalg.norm(np.array(query) - np.array(key))
            best_match_key = key
    return best_match_key


def match_apparel(query, key):
    print(f"query {query}")
    query = get_match(query)
    print(f"match color {key}")
    key = get_match(key)
    if key in palette_rgb[query]:
        print("good match")
        return True
    else:
        print("not good match")
        return False


if __name__ == "__main__":
    query = (174,182, 200)
    key = (0,0,0)
    match_apparel(query, key)