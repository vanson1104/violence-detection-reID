from fastapi import FastAPI, APIRouter, HTTPException, status
from .schemas import RequestData
from .utils import process_request_data
from ai_engine import extract_id_engine

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "ok"}

@router.post("/predict", status_code=status.HTTP_200_OK) 
async def predict(data: RequestData): 
    image = process_request_data(data)
    results = await extract_id_engine(image)
    return results


