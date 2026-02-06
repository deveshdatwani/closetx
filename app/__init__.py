from fastapi import FastAPI

def create_app():
    app = FastAPI()
    from app.routes.closet import closet_router
    app.include_router(closet_router) 
    return app

