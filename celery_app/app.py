from flask import Flask, request, send_file
from PIL import Image
from io import BytesIO
import os, io
from celery_back_end import run_inference, get_result
from base64 import encodebytes


def create_app(config_file=None): 
    app = Flask(__name__)
    return app


app = create_app()


@app.route('/')
def infer():
    img = request.files['image']
    img.save("new_image.png")
    result = run_inference.apply_async(args=["new_image.png"])
    return str(type(result))


@app.route('/task/<task_id>')
def get_task(task_id):
    img_path = get_result(task_id)
    apparel_image = Image.open(img_path)
    img_io = io.BytesIO()
    apparel_image.save(img_io, 'PNG')
    img_io.seek(0) 
    return send_file(img_io, mimetype='image/png')
