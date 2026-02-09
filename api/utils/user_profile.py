import mysql.connector, os, logging, jwt, datetime, hashlib
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("user_app")
SECRET_KEY = os.getenv("JWT_SECRET", "supersecret")
ALGORITHM = "HS256"

DB_CONFIG = {"host": os.getenv("DB_HOST","localhost"),"user":os.getenv("DB_USER","user"),"password":os.getenv("DB_PASS","pass"),"database":os.getenv("DB_NAME","closetx")}

def get_conn():
    logger.info("Creating DB connection")
    return mysql.connector.connect(**DB_CONFIG)

def hash_password(password: str):
    return hashlib.sha256(password.encode()).hexdigest()

def create_jwt(user_id: int):
    payload = {"user_id": user_id,"exp": datetime.datetime.utcnow()+datetime.timedelta(hours=1)}
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"JWT created for user {user_id}")
    return token

def verify_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["user_id"]
    except Exception as e:
        logger.error(f"JWT verification failed: {e}")
        return None

def create_user_db(username: str, password: str):
    conn = get_conn()
    cursor = conn.cursor()
    hashed = hash_password(password)
    try:
        cursor.execute("INSERT INTO user (username,password) VALUES (%s,%s)", (username, hashed))
        conn.commit()
        user_id = cursor.lastrowid
        token = create_jwt(user_id)
        logger.info(f"User created: {username} with id {user_id}")
        return {"id": user_id, "username": username, "token": token}
    except mysql.connector.Error as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=400, detail="User creation failed")
    finally:
        cursor.close()
        conn.close()

def login_user_db(username: str, password: str):
    conn = get_conn()
    cursor = conn.cursor(dictionary=True)
    hashed = hash_password(password)
    cursor.execute("SELECT * FROM user WHERE username=%s AND password=%s",(username,hashed))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if not user:
        logger.warning(f"Failed login attempt for {username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_jwt(user["id"])
    logger.info(f"User logged in: {username}")
    return {"id": user["id"], "username": user["username"], "token": token, "access_token": token}

def edit_user_db(user_id: int, username: str = None, password: str = None):
    if not username and not password:
        raise HTTPException(status_code=400, detail="Nothing to update")
    conn = get_conn()
    cursor = conn.cursor()
    updates = []
    values = []
    if username:
        updates.append("username=%s")
        values.append(username)
    if password:
        updates.append("password=%s")
        values.append(hash_password(password))
    values.append(user_id)
    try:
        cursor.execute(f"UPDATE user SET {','.join(updates)} WHERE id=%s", tuple(values))
        conn.commit()
        logger.info(f"User updated: {user_id}")
        return {"status":"updated"}
    except mysql.connector.Error as e:
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(status_code=400, detail="Update failed")
    finally:
        cursor.close()
        conn.close()

def delete_user_db(user_id: int):
    conn = get_conn()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM user WHERE id=%s", (user_id,))
        conn.commit()
        logger.info(f"User deleted: {user_id}")
        return {"status":"deleted"}
    except mysql.connector.Error as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=400, detail="Delete failed")
    finally:
        cursor.close()
        conn.close()