from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
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

# Known width of the object (e.g., cigarette box) in cm
BOX_WIDTH = 5.4
focal_length = None

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

def detect_object(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    image_np = np.array(image)

    # Convert to grayscale
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        return image_np, (x, y, w, h)
    return None, None

@app.post("/calibrate")
async def calibrate(known_distance: float = Form(...), file: UploadFile = File(...)):
    global focal_length
    image_bytes = await file.read()

    # Detect object and get its width
    image_np, bounding_box = detect_object(image_bytes)
    if bounding_box:
        x, y, w, h = bounding_box
        focal_length = calculate_focal_length(known_distance, BOX_WIDTH, w)
        return JSONResponse({"success": True, "focal_length": focal_length})
    else:
        return JSONResponse({"success": False, "message": "Object not detected"})

@app.post("/measure")
async def measure(file: UploadFile = File(...)):
    global focal_length
    if focal_length is None:
        return JSONResponse({"success": False, "message": "Camera not calibrated"})

    image_bytes = await file.read()

    # Detect object and measure distance
    image_np, bounding_box = detect_object(image_bytes)
    if bounding_box:
        x, y, w, h = bounding_box
        distance = estimate_distance(focal_length, BOX_WIDTH, w)

        # Draw bounding box and distance on the image
        cv2.rectangle(image_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image_np, f"Distance: {distance:.2f} cm", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Convert back to bytes for display
        _, buffer = cv2.imencode('.jpg', image_np)
        return JSONResponse({"success": True, "distance": distance})
    else:
        return JSONResponse({"success": False, "message": "Object not detected"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
