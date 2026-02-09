from fastapi import APIRouter, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from uuid import uuid4
import os, shutil
from celery import Celery
from api.utils.cache import raw_path, proc_path
from api.utils.db import get_conn
from api.utils.s3 import upload as s3_upload, delete as s3_delete
from shared.task_contracts import SEGMENT_TASK
router = APIRouter()
celery = Celery(broker=os.getenv("CELERY_BROKER"))

@router.post("/upload")
async def upload_image(file: UploadFile, user: int = Form(...)):
    print("uploaded image")
    uri = str(uuid4())
    with open(raw_path(uri), "wb") as f:
        shutil.copyfileobj(file.file, f)
    #s3_upload(raw_path(uri), f"raw/{uri}.png")
    db = get_conn()
    cur = db.cursor()
    cur.execute("INSERT INTO apparel(user, uri) VALUES (%s,%s)", (user, uri))
    db.commit()
    #celery.send_task(SEGMENT_TASK, args=[uri])
    return {"uri": uri}

@router.get("/list")
def list_images(user: int):
    db = get_conn()
    cur = db.cursor()
    cur.execute("SELECT uri FROM apparel WHERE user=%s", (user,))
    return [r[0] for r in cur.fetchall()]

@router.get("/fetch/{uri}")
def fetch_image(uri: str):
    if os.path.exists(proc_path(uri)):
        return FileResponse(proc_path(uri))
    if os.path.exists(raw_path(uri)):
        return FileResponse(raw_path(uri))
    raise HTTPException(404, "image not found")

@router.delete("/{uri}")
def delete_image(uri: str):
    s3_delete(f"raw/{uri}.png")
    s3_delete(f"processed/{uri}.png")
    for p in [raw_path(uri), proc_path(uri)]:
        if os.path.exists(p):
            os.remove(p)
    db = get_conn()
    cur = db.cursor()
    cur.execute("DELETE FROM apparel WHERE uri=%s", (uri,))
    db.commit()
    return {"deleted": uri}

