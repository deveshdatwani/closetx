import celery


@celery.Task
def run_inference(img):
    result = "something"
    return result