from fastapi import FastAPI
import os, logging
from api.routes.images import router as images_router
from api.routes.user import router as user_router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(images_router, prefix="/images")
app.include_router(user_router, prefix="/user")
logger.info("app_initialized", extra={"prefix": ["/images", "/user"]})