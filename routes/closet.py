from flask import Blueprint, request, jsonify
import mysql.connector, os


images_bp = Blueprint('images', __name__)


@images_bp.route('/user/<int:user_id>', methods=['GET'])
def get_user_images(user_id):
    conn = mysql.connector.connect(host="localhost", user="user", password="pass", database="db")
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM images WHERE user_id=%s", (user_id,))
    images = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(images)

@images_bp.route('/upload', methods=['POST'])
def upload_image():
    user_id = request.form['user_id']
    file = request.files['image']
    filename = file.filename
    save_path = os.path.join("/absolute/path/to/save", filename)
    file.save(save_path)
    conn = mysql.connector.connect(host="localhost", user="user", password="pass", database="db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO images (user_id, file_path, file_name) VALUES (%s, %s, %s)", (user_id, save_path, filename))
    conn.commit()
    cursor.close()
    conn.close()
    return jsonify({"status":"success","filename":filename})

