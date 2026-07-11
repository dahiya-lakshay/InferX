from fastapi import FastAPI

app = FastAPI(
    title="InferX",
    version="1.0.0"
)


@app.get("/")
def root():
    return {
        "message": "InferX Backend Running"
    }