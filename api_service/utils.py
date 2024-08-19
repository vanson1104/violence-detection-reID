import base64
import numpy as np
from io import BytesIO
from PIL import Image

def process_request_data(sample):
    decoded_bytes = base64.b64decode(sample.image)
    image = Image.open(BytesIO(decoded_bytes)).convert("RGB")
    image = np.array(image)
    return image
