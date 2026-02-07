import os, logging
from fastapi import FastAPI
from dotenv import dotenv_values, load_dotenv

def create_app():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    for k, v in dotenv_values().items():
        logger.info(f"{k}={v}")
    app = FastAPI()
    from closetx.app.routes.closet import closet_router
    app.include_router(closet_router)
    return app