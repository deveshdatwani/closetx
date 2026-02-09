from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel
from utils.user_profile import create_user_db, login_user_db, edit_user_db, delete_user_db, verify_jwt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("user_routes")

router = APIRouter(prefix="/user")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/user/login")

class UserCreate(BaseModel):
    username: str
    password: str

class UserEdit(BaseModel):
    username: str = None
    password: str = None

@router.post("/create")
def create_user(user: UserCreate):
    logger.info(f"Received create request for username: {user.username}")
    result = create_user_db(user.username, user.password)
    logger.info(f"User created: {result['id']}")
    return result

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    logger.info(f"Login attempt for username: {form_data.username}")
    result = login_user_db(form_data.username, form_data.password)
    logger.info(f"Login successful for username: {form_data.username}")
    return result

@router.put("/edit/{user_id}")
def edit_user(user_id: int, user: UserEdit, token: str = Depends(oauth2_scheme)):
    auth_id = verify_jwt(token)
    logger.info(f"Edit request for user_id: {user_id} by auth_id: {auth_id}")
    if auth_id != user_id:
        logger.warning(f"Unauthorized edit attempt by {auth_id} on {user_id}")
        raise HTTPException(status_code=403, detail="Forbidden")
    result = edit_user_db(user_id, user.username, user.password)
    logger.info(f"User updated: {user_id}")
    return result

@router.delete("/delete/{user_id}")
def delete_user(user_id: int, token: str = Depends(oauth2_scheme)):
    auth_id = verify_jwt(token)
    logger.info(f"Delete request for user_id: {user_id} by auth_id: {auth_id}")
    if auth_id != user_id:
        logger.warning(f"Unauthorized delete attempt by {auth_id} on {user_id}")
        raise HTTPException(status_code=403, detail="Forbidden")
    result = delete_user_db(user_id)
    logger.info(f"User deleted: {user_id}")
    return result