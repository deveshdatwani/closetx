from fastapi import APIRouter, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from uuid import uuid4
import os, shutil, logging
from celery import Celery
from api.utils.cache import raw_path, proc_path
from api.utils.db import get_conn
from api.utils.errors import api_error_handler
from api.utils.s3 import upload as s3_upload, delete as s3_delete

router = APIRouter()

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

CELERY_BROKER = os.getenv("CELERY_BROKER")
ENABLE_S3 = os.getenv("ENABLE_S3", "false").lower() == "true"
ENABLE_CELERY = os.getenv("ENABLE_CELERY", "false").lower() == "true"
SEGMENT_TASK = os.getenv("SEGMENT_TASK")

celery = Celery(broker=CELERY_BROKER) if CELERY_BROKER else None

@router.post("/upload")
@api_error_handler
async def upload_image(file: UploadFile, user: int = Form(...)):
    logger.info("upload_start", extra={"user": user})
    # validate user exists to avoid foreign key integrity errors
    db = get_conn()
    cur = db.cursor()
    cur.execute("SELECT id FROM `user` WHERE id=%s", (user,))
    if cur.fetchone() is None:
        logger.error("upload_user_not_found", extra={"user": user})
        raise HTTPException(status_code=400, detail="user not found")
    uri = str(uuid4())
    path = raw_path(uri)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("file_saved", extra={"uri": uri, "path": path})
    if ENABLE_S3:
        s3_upload(path, f"raw/{uri}.png")
        logger.info("s3_uploaded", extra={"uri": uri})
    try:
        cur.execute("INSERT INTO apparel(user, uri) VALUES (%s,%s)", (user, uri))
        db.commit()
    except Exception:
        logger.exception("db_insert_failed", extra={"user": user, "uri": uri})
        raise HTTPException(500, "failed to save image record")
    logger.info("db_inserted", extra={"uri": uri, "user": user})
    if ENABLE_CELERY and celery and SEGMENT_TASK:
        celery.send_task(SEGMENT_TASK, args=[uri])
        logger.info("celery_task_sent", extra={"uri": uri})
    return {"uri": uri}

@router.get("/list")
@api_error_handler
def list_images(user: int):
    logger.info("list_request", extra={"user": user})
    db = get_conn()
    cur = db.cursor()
    cur.execute("SELECT uri FROM apparel WHERE user=%s", (user,))
    rows = [r[0] for r in cur.fetchall()]
    logger.info("list_response", extra={"user": user, "count": len(rows)})
    return rows

@router.get("/fetch/{uri}")
@api_error_handler
def fetch_image(uri: str):
    logger.info("fetch_request", extra={"uri": uri})
    proc = proc_path(uri)
    raw = raw_path(uri)
    if os.path.exists(proc):
        logger.info("fetch_processed", extra={"uri": uri})
        return FileResponse(proc)
    if os.path.exists(raw):
        logger.info("fetch_raw", extra={"uri": uri})
        return FileResponse(raw)
    logger.error("fetch_not_found", extra={"uri": uri})
    raise HTTPException(404, "image not found")

@router.delete("/{uri}")
@api_error_handler
def delete_image(uri: str):
    logger.info("delete_start", extra={"uri": uri})
    if ENABLE_S3:
        s3_delete(f"raw/{uri}.png")
        s3_delete(f"processed/{uri}.png")
        logger.info("s3_deleted", extra={"uri": uri})
    for p in [raw_path(uri), proc_path(uri)]:
        if os.path.exists(p):
            os.remove(p)
            logger.info("file_deleted", extra={"path": p})
    db = get_conn()
    cur = db.cursor()
    cur.execute("DELETE FROM apparel WHERE uri=%s", (uri,))
    db.commit()
    logger.info("db_deleted", extra={"uri": uri})
    return {"deleted": uri}