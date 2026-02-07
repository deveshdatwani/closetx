from fastapi import APIRouter, UploadFile, Form, HTTPException, Form
from uuid import uuid4
from fastapi.responses import FileResponse, JSONResponse, Response
import mysql.connector, os, shutil
from closetx.app.utils.db import get_conn
import os, shutil
from model.tasks import add_two_numbers
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.Logger(__name__)
logger.setLevel(logging.INFO)

UPLOAD_DIR = os.environ.get("UPLOAD_DIR","/absolute/path/to/save")
closet_router = APIRouter()

@closet_router.get("/")
def get_home():
    add_two_numbers.delay(random.randint(1, 100), 5)
    return {"status":"success", "data":"welcome to your closet"}

@closet_router.get("/user/{user_id}")
def get_user_images(user_id: int):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT uri FROM apparel WHERE user=%s", (user_id,))
        all_image_uri = cursor.fetchall()
        cursor.close()
        conn.close()
        return all_image_uri
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@closet_router.post("/upload")
def upload_image(user: int = Form(...), image: UploadFile = Form(...)):
    try:
        uri = str(uuid4())
        raw_dir = os.path.join(UPLOAD_DIR, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        save_path = os.path.join(raw_dir, f"{uri}.png")
        with open(save_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO apparel (user, uri) VALUES (%s, %s)", (user, uri))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status": "success", "uri": uri}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@closet_router.delete("/delete/{image_id}")
def delete_image(image_id: int):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT file_path FROM images WHERE id=%s", (image_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="image not found")
        file_path = row['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
        cursor.execute("DELETE FROM images WHERE id=%s", (image_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status":"success","deleted_id":image_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@closet_router.get("/fetch/{uri}")
def fetch_image(uri: str):
    try:
        file_path = os.path.join(UPLOAD_DIR, "processed", f"{uri}.png")
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="image not found")
        with open(file_path, "rb") as f:
            img_bytes = f.read()
        return Response(
            content=img_bytes,
            media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))