from fastapi import FastAPI

from app.api.routes import router

from app.core.bootstrap import bootstrap

app = FastAPI(
    title="InferX",
    version="1.0.0",
)

@app.on_event("startup")
def startup() -> None:
    bootstrap()

app.include_router(router)

@app.get("/")
def root():
    return {
        "message": "InferX Backend Running"
    }

