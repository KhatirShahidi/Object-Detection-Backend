import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import io

app = FastAPI()

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
BOX_WIDTH = 5.4  # Width of the reference object (cigarette box) in cm
KNOWN_DISTANCE = 30.0  # Known distance in cm for calibration

# Utility functions
def calculate_focal_length(known_distance, known_width, perceived_width):
    return (perceived_width * known_distance) / known_width

def estimate_distance(focal_length, known_width, perceived_width):
    return (known_width * focal_length) / perceived_width

def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return image

def detect_object(image, focal_length, known_width):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        distance = estimate_distance(focal_length, known_width, w)
        
        # Draw the bounding box and annotate distance
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return image, distance
    return None, None

@app.post("/api/calibrate")
async def calibrate(file: UploadFile = File(...)):
    # Read the uploaded file
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Detect the reference object and calculate focal length
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        focal_length = calculate_focal_length(KNOWN_DISTANCE, BOX_WIDTH, w)
        return {"success": True, "focal_length": focal_length}
    else:
        return {"success": False, "message": "Object not detected"}

@app.post("/api/measure")
async def measure(file: UploadFile = File(...), focal_length: float = 0):
    # Read the uploaded file
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    # Detect the object and measure distance
    processed_image, distance = detect_object(image, focal_length, BOX_WIDTH)
    if processed_image is not None:
        _, img_encoded = cv2.imencode('.jpg', processed_image)
        return {
            "success": True,
            "distance": distance,
            "image": img_encoded.tobytes()
        }
    else:
        return {"success": False, "message": "Object not detected"}

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
