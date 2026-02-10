from fastapi import FastAPI
import os, logging
from api.routes.images import router as images_router
from api.routes.user import router as user_router
from api.routes.inference import router as inference_router
from fastapi.responses import JSONResponse
from fastapi import Request
from api.utils.errors import api_error_handler
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)
app = FastAPI()
app.include_router(images_router, prefix="/images")
app.include_router(user_router, prefix="/user")
app.include_router(inference_router, prefix="/inference")
logger.info("app_initialized", extra={"prefix": ["/images", "/user", "/inference"]})

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
	logging.getLogger(__name__).exception("unhandled_exception", exc_info=exc)
	return JSONResponse(status_code=500, content={"detail": "internal server error"})