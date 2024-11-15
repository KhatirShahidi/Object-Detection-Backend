from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import cv2
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BOX_WIDTH = 5.4  # width in cm
KNOWN_DISTANCE = 30  # in cm

def calculate_focal_length(known_distance, known_width, perceived_width):
    return (perceived_width * known_distance) / known_width

def estimate_distance(focal_length, known_width, perceived_width):
    return (known_width * focal_length) / perceived_width

def detect_object(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            return image_np, (x, y, w, h)
    except Exception as e:
        print(f"Error in detect_object: {e}")
    return None, None

@app.post("/api/calibrate")
async def calibrate(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid file type")

        image_bytes = await file.read()
        image_np, bounding_box = detect_object(image_bytes)
        if bounding_box:
            x, y, w, h = bounding_box
            focal_length = calculate_focal_length(KNOWN_DISTANCE, BOX_WIDTH, w)
            return {"success": True, "focal_length": focal_length}
        return {"success": False, "message": "Object not detected"}
    except Exception as e:
        print(f"Error in calibrate: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/api/measure")
async def measure(
    file: UploadFile = File(...), focal_length: float = Form(...)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    image_bytes = await file.read()
    image_np, bounding_box = detect_object(image_bytes)
    if bounding_box:
        x, y, w, h = bounding_box
        distance = estimate_distance(focal_length, BOX_WIDTH, w)
        return {"success": True, "distance": distance}
    return {"success": False, "message": "Object not detected"}
