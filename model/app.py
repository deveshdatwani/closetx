from celery import Celery
from models.huggingface_cloth_segmentation.process import main
from models.huggingface_cloth_segmentation.process import make_model


celery = Celery("ml_app",
                broker="redis://localhost:6379/0",
                backend="redis://localhost:6379/1")


segmentation_model = make_model()


@celery.task
def segment(image):
    device = "cpu"
    main(image, device, segmentation_model)