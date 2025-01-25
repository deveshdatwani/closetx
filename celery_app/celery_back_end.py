from celery import Celery
from PIL import Image
from time import sleep
from celery.result import AsyncResult


celery_app = Celery(__name__,
                    broker='redis://localhost:6379/0',
                    backend='redis://localhost:6379/1'
                    )


@celery_app.task()
def run_inference(img_name):
    result = Image.open(img_name).resize((300,300))
    new_img_name = f"new_image_{img_name}"
    result.save(new_img_name)
    return new_img_name


@celery_app.task()
def get_result(task_id):
    task = AsyncResult(task_id, app=celery_app)
    img = task.get()
    return img


if __name__ == "__main__":
    celery_app.start()