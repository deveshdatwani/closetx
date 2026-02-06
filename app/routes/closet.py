from fastapi import APIRouter, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import mysql.connector, os, shutil
from app.utils.db import get_conn

UPLOAD_DIR = os.environ.get("UPLOAD_DIR","/absolute/path/to/save")
closet_router = APIRouter()

@closet_router.get("/")
def get_home():
    return {"status":"success", "data":"welcome to your closet"}


@closet_router.get("/user/{user_id}")
def get_user_images(user_id: int):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM images WHERE user_id=%s", (user_id,))
        images = cursor.fetchall()
        cursor.close()
        conn.close()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@closet_router.post("/upload")
def upload_image(user_id: int = Form(...), image: UploadFile = Form(...)):
    try:
        filename = image.filename
        save_path = os.path.join(UPLOAD_DIR, filename)
        with open(save_path, "wb") as f:
            shutil.copyfileobj(image.file, f)
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (user_id, file_path, file_name) VALUES (%s,%s,%s)", (user_id, save_path, filename))
        conn.commit()
        cursor.close()
        conn.close()
        return {"status":"success","filename":filename}
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

@closet_router.get("/fetch/{image_id}")
def fetch_image(image_id: int):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT file_path FROM images WHERE id=%s", (image_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row or not os.path.exists(row['file_path']):
            raise HTTPException(status_code=404, detail="image not found")
        return FileResponse(row['file_path'], filename=os.path.basename(row['file_path']))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

