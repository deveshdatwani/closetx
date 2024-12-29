import process
import config
import os
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
from PIL import Image


class DatasetCreater():
    def __init__(self, config_file=None):
        self.load_config(config_file)
        print(f"loaded raw image {self.raw}")
        print(f"loaded top {self.positive_top}")
        print(f"loaded bottom {self.positive_bottom}")

    def load_config(self, config_file):
        print("loading config")
        self.raw = config.RAW
        self.positive_top = config.POSITIVE_TOP
        self.positive_bottom = config.POSITIVE_BOTTOM
    
    def curate_dataset(self):
        model = process.get_model()
        raw_images = os.listdir(self.raw)
        count = 0
        for image in raw_images:
            count += 1
            image_path = os.path.join(self.raw, image)
            print(f"segmenting {image} || number: {count}")
            top, bottom = process.segment_apparel(image_path, 'cpu', model)
            top_image_fp = os.path.join(self.positive_top, image)
            bottom_image_fp = os.path.join(self.positive_bottom, image)
            top.save(top_image_fp)
            bottom.save(bottom_image_fp)


creater = DatasetCreater()
creater.curate_dataset()