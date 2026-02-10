from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os, logging
from uuid import uuid4
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from celery import Celery
from api.utils.db import get_conn
from api.utils.cache import save_raw_bytes, record_source_mapping

router = APIRouter()
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
CELERY_BROKER = os.getenv("CELERY_BROKER")
ENABLE_CELERY = os.getenv("ENABLE_CELERY", "false").lower() == "true"
celery = Celery(broker=CELERY_BROKER) if CELERY_BROKER else None

class MatchRequest(BaseModel):
    user: Optional[int] = None
    dashboard_uris: Optional[List[str]] = None
    page_images: List[str]
    text: Optional[str] = "does this apparel I have in closet go with the apparel in the online store"

@router.post("/match")
def match(req: MatchRequest):
    if not req.dashboard_uris:
        if req.user is None:
            raise HTTPException(status_code=400, detail="user or dashboard_uris required")
        db = get_conn()
        cur = db.cursor()
        cur.execute("SELECT uri FROM apparel WHERE user=%s", (req.user,))
        req.dashboard_uris = [r[0] for r in cur.fetchall()]
    if not CELERY_BROKER:
        raise HTTPException(status_code=500, detail="CELERY_BROKER not configured")
    processed_page_uris: List[str] = []
    page_uri_map = {}
    for img in req.page_images:
        if isinstance(img, str) and img.lower().startswith("http"):
            try:
                req = Request(img, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "image/*,*/*;q=0.8"})
                resp = urlopen(req, timeout=15)
                data = resp.read()
                ctype = None
                try:
                    ctype = resp.headers.get_content_type()
                except Exception:
                    ctype = None
            except (HTTPError, URLError, TimeoutError) as e:
                logger.error("page_image_fetch_failed", extra={"url": img, "error": str(e)})
                raise HTTPException(status_code=400, detail=f"failed to fetch image {img}: {e}")
            ext_map = {"image/jpeg": "jpg", "image/jpg": "jpg", "image/png": "png", "image/gif": "gif", "image/webp": "webp"}
            ext = ext_map.get(ctype, None)
            if not ext:
                _, url_ext = os.path.splitext(img)
                if url_ext:
                    ext = url_ext.lstrip('.')
                else:
                    ext = None
            gen_uri = f"page_{uuid4().hex}"
            save_raw_bytes(gen_uri, data, ext)
            record_source_mapping(gen_uri, img)
            processed_page_uris.append(gen_uri)
            page_uri_map[gen_uri] = img
        else:
            processed_page_uris.append(img)
            page_uri_map[img] = img
    res = celery.send_task("closetx.worker.task.match_images_batch", args=[req.dashboard_uris, processed_page_uris, req.text])
    queued_pairs = len(req.dashboard_uris) * len(processed_page_uris)
    logger.info("match_batch_queued", extra={"task_id": res.id, "pairs": queued_pairs})
    return {"queued_pairs": queued_pairs, "task_id": res.id, "page_uri_map": page_uri_map}

@router.get("/result/{task_id}")
def result(task_id: str):
    if not CELERY_BROKER:
        raise HTTPException(status_code=500, detail="CELERY_BROKER not configured")
    res = celery.AsyncResult(task_id)
    if not res:
        raise HTTPException(status_code=404, detail="task not found")
    if not res.ready():
        return {"state": res.state}
    return {"state": res.state, "result": res.result}