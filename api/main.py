from fastapi import FastAPI
import os, logging
from api.routes.images import router

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

APP_PREFIX = os.getenv("APP_PREFIX", "/images")

app = FastAPI()
app.include_router(router, prefix=APP_PREFIX)
logger.info("app_initialized", extra={"prefix": APP_PREFIX})