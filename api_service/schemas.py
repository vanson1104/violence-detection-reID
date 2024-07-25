from pydantic import BaseModel, field_validator
import re

class RequestData(BaseModel):
    image : str

    # @field_validator('image')
    # def check_image(cls, value):
    #     if not len(value):
    #         raise ValueError("Empty payload")
    #     # Check for valid base64 format using regex
    #     if not re.match(
    #         r"^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$", value
    #     ):
    #         raise ValueError("Invalid base64 image format")
    #     return value