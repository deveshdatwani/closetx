from fastapi import FastAPI
from api.routes.images import router
app = FastAPI()
app.include_router(router, prefix="/images")

