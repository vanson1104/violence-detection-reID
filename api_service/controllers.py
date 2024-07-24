from fastapi import FastAPI, APIRouter, HTTPException, status
from .schemas import RequestData

router = APIRouter()

@router.get("/")
async def health_check():
    return {"status": "ok"}

@router.post("/predict", status_code=status.HTTP_200_OK) 
async def predict(request_data: RequestData):    
    return {"prediction": "dog"}


