from uvicorn import run 
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_service import violence_router

def _get_app():
    app = FastAPI(docs_url="/docs")

    origins = [
    "http://localhost:5050",
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost"
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(violence_router, tags=["VIOLENCE DETECTION API"])
    return app

app = _get_app()

if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8686, reload=True)

