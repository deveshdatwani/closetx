import os
from celery import Celery
from PIL import Image
from model.models.huggingface_cloth_segmentation.process import main as segment_apparel, get_model
import logging

logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

model = get_model()
HOST = os.getenv("HOST", "localhost") 

logger.info(f"redis host set to {HOST}")

app = Celery("flask",
             broker=f"redis://{HOST}:6379/0",
             backend=f"redis://{HOST}:6379/0")

@app.task(name="tasks.infer")
def segment_apparel_task(image_path):
    image = Image.open(image_path)
    image = segment_apparel(image=image, model=model)
    image.save(image_path)
    return True