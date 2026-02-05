from flask import Blueprint, request, jsonify, send_file
import mysql.connector, os

UPLOAD_DIR = os.environ.get("UPLOAD_DIR","/absolute/path/to/save")

@images_bp.route('/user/<int:user_id>', methods=['GET'])
def get_user_images(user_id):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM images WHERE user_id=%s", (user_id,))
        images = cursor.fetchall()
        cursor.close()
        conn.close()
        return jsonify(images)
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

@images_bp.route('/upload', methods=['POST'])
def upload_image():
    try:
        user_id = request.form['user_id']
        file = request.files['image']
        filename = file.filename
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO images (user_id, file_path, file_name) VALUES (%s, %s, %s)", (user_id, save_path, filename))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status":"success","filename":filename})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

@images_bp.route('/delete/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT file_path FROM images WHERE id=%s", (image_id,))
        row = cursor.fetchone()
        if not row:
            cursor.close()
            conn.close()
            return jsonify({"status":"error","message":"image not found"}),404
        file_path = row['file_path']
        if os.path.exists(file_path):
            os.remove(file_path)
        cursor.execute("DELETE FROM images WHERE id=%s", (image_id,))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"status":"success","deleted_id":image_id})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

@images_bp.route('/fetch/<int:image_id>', methods=['GET'])
def fetch_image(image_id):
    try:
        conn = get_conn()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT file_path FROM images WHERE id=%s", (image_id,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        if not row or not os.path.exists(row['file_path']):
            return jsonify({"status":"error","message":"image not found"}),404
        return send_file(row['file_path'], as_attachment=True)
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}),500

