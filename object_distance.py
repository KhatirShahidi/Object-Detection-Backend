import cv2
import numpy as np

def calculate_focal_length(known_distance, known_width, perceived_width):
    """
    Calculate the focal length using a known distance and object width.
    """
    return (perceived_width * known_distance) / known_width

def estimate_distance(focal_length, known_width, perceived_width):
    """
    Estimate the distance of an object using the perceived width in the image.
    """
    return (known_width * focal_length) / perceived_width

def resize_image(image, max_width=800, max_height=600):
    """
    Resize the image for display, while keeping the aspect ratio.
    """
    height, width = image.shape[:2]
    if width > max_width or height > max_height:
        scaling_factor = min(max_width / width, max_height / height)
        new_size = (int(width * scaling_factor), int(height * scaling_factor))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        return resized_image
    return image

def detect_object_and_measure_distance(image_path, known_width, focal_length):
    """
    Detect a reference object (cigarette box) in an image and estimate its distance.
    """
    # Load the original image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError("Image not found or unable to load.")
    
    # Convert the original image to grayscale for processing
    gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny edge detector
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort by area and get the largest contour
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    if contours:
        # Get the bounding box of the largest contour
        x, y, w, h = cv2.boundingRect(contours[0])
        
        # Estimate the distance using the width in the original image
        distance = estimate_distance(focal_length, known_width, w)
        
        # Draw the bounding box and distance on a resized image for display
        display_image = resize_image(original_image)
        scale_factor = display_image.shape[1] / original_image.shape[1]
        cv2.rectangle(display_image, (int(x * scale_factor), int(y * scale_factor)), 
                      (int((x + w) * scale_factor), int((y + h) * scale_factor)), (0, 255, 0), 2)
        
        # Display the estimated distance on the resized image
        cv2.putText(display_image, f"Distance: {distance:.2f} cm", (int(x * scale_factor), int((y - 10) * scale_factor)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show the resized image
        cv2.imshow("Detected Object", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return distance
    else:
        print("No object detected")
        return None

# Example usage:
# Cigarette box dimensions (fixed, known values in cm)
BOX_WIDTH = 5.4  # width of the cigarette box in cm

# Calibration: known distance and perceived width of the box in pixels (from a calibration image)
calibration_image_path = "calibration_image.jpg"  # Replace with your calibration image path
KNOWN_DISTANCE = 30.0  # known distance in cm

# Load the calibration image
calibration_image = cv2.imread(calibration_image_path)
gray_calib = cv2.cvtColor(calibration_image, cv2.COLOR_BGR2GRAY)
edges_calib = cv2.Canny(gray_calib, 50, 150)
contours_calib, _ = cv2.findContours(edges_calib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the width of the cigarette box in pixels from the calibration image
contours_calib = sorted(contours_calib, key=cv2.contourArea, reverse=True)
x, y, calib_width, h = cv2.boundingRect(contours_calib[0])

# Calculate focal length
focal_length = calculate_focal_length(KNOWN_DISTANCE, BOX_WIDTH, calib_width)

# Estimate distance using a new image
test_image_path = "test_image.jpg"  # Replace with your test image path
distance = detect_object_and_measure_distance(test_image_path, BOX_WIDTH, focal_length)

if distance:
    print(f"Estimated Distance: {distance:.2f} cm")
