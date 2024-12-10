import os 
import sys
from PIL import Image
from process import segment_apparel


def generate(datapath="./dataset/"):
    try:
        os.path.exists(datapath)
    except Exception as e:
        print(e)
        return
    if not os.path.exists(os.path.join(datapath, "top")):
        top_path = os.mkdir(os.path.join(datapath, "top"))
    if not os.path.exists(os.path.join(datapath, "bottom")):
        bottom_path = os.mkdir(os.path.join(datapath, "bottom"))
    for i, image_path in enumerate(os.listdir(path=datapath)):
        apparel = Image.open(image_path)
        mask, top, bottom = segment_apparel(apparel)
        apparel_top = Image.new("RGB", apparel.size)
        apparel_bottom = Image.new("RGB", apparel.size)
        apparel_top.paste(apparel, mask=top)
        apparel_bottom.paste(apparel, mask=bottom)
        
        with open(os.path.join(datapath, "top"), 'rb') as file:
            file.save(apparel_top, f"{i}.jpg")
        break
    return f"Successfully generated {i+1} samples"


if __name__ == "__main__":
    dataset_path = "/home/deveshdatwani/closetx/models/dataset/v1/raw"
    generate(datapath=dataset_path)