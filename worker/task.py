import logging
import random
from typing import Any, Dict, List
from .celery_app import app
logger = logging.getLogger(__name__)

from api.utils.errors import task_error_wrapper


@app.task(name='closetx.worker.task.match_images')
@task_error_wrapper
def match_images(dashboard_uri: str, page_image_url: str, text: str = "does this apparel I have in closet go with the apparel in the online store") -> Dict[str, Any]:
    logger.info("match_images_task_start", extra={"dashboard_uri": dashboard_uri, "page_image": page_image_url})
    # Temporary heuristic: random score and highlight threshold
    score = round(random.uniform(0.0, 1.0), 3)
    highlight = score >= 0.7
    result = {"dashboard_uri": dashboard_uri, "page_image_url": page_image_url, "text": text, "score": score, "highlight": highlight}
    logger.info("match_images_task_done", extra=result)
    return result


@app.task(name='closetx.worker.task.match_images_batch')
@task_error_wrapper
def match_images_batch(dashboard_uris: List[str], page_image_urls: List[str], text: str = "does this t shirt I have in closet go with the apparel in the online store") -> List[Dict[str, Any]]:
    logger.info("match_images_batch_start", extra={"dash_count": len(dashboard_uris), "page_count": len(page_image_urls)})
    results = []
    for d in dashboard_uris:
        for p in page_image_urls:
            score = round(random.uniform(0.0, 1.0), 3)
            highlight = score >= 0.7
            results.append({"dashboard_uri": d, "page_image_url": p, "text": text, "score": score, "highlight": highlight})
    logger.info("match_images_batch_done", extra={"pairs": len(results)})
    return results