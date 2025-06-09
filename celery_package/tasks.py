import time
import joblib
from app import celery
from models.huggingface_cloth_segmentation.process import make_model


segmentation_model = make_model()


@celery.task
def semgent():
    # something 
    return "Done"