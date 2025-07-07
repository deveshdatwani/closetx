from model.models.huggingface_cloth_segmentation.process import main as infer
from PIL import Image
import numpy as np


img_path = "/home/deveshdatwani/Pictures/old-navy.png"
img = Image.open(img_path)
img = img.resize((786, 786))
seg_img = infer(img)
seg_img.show()