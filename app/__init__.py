from fastapi import FastAPI
from dotenv import load_dotenv

def create_app():
    load_dotenv()
    app = FastAPI()
    from app.routes.closet import closet_router
    app.include_router(closet_router) 
    return app

