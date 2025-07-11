import cv2 #type: ignore
import numpy as np

def calculate_black_ratio(frame, x1, y1, x2, y2):
    """
    Calculate the ratio of black pixels in the torso region of a person in a frame.
    Used to determine if a person is wearing a black shirt (e.g., employee vs. customer).
    
    Args:
        frame (np.ndarray): Input video frame in BGR format.
        x1, y1, x2, y2 (float): Bounding box coordinates for the person.
    
    Returns:
        float: Ratio of black pixels in the torso region (0 to 1).
               Returns -1 if the ROI is invalid or empty.
    """
    # Extract the region of interest (ROI) for the person
    person_roi = frame[int(y1):int(y2), int(x1):int(x2)]
    if person_roi.size == 0:
        return -1

    # Convert ROI to HSV for color-based detection
    hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)

    # Apply CLAHE to Value channel for brightness normalization
    h, s, v = cv2.split(hsv_roi)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)
    hsv_eq = cv2.merge([h, s, v_eq])

    # Focus on the center torso region (30% to 60% height)
    height, width = hsv_eq.shape[:2]
    start_y = int(height * 0.3)
    end_y = int(height * 0.6)
    crop_margin = int(width * 0.2)
    upper_body_hsv = hsv_eq[start_y:end_y, crop_margin:width - crop_margin]
    
    # Convert to BGR for consistency (optional, kept for compatibility)
    upper_body_bgr = cv2.cvtColor(upper_body_hsv, cv2.COLOR_HSV2BGR)

    # Define relaxed black HSV range to exclude black shirts
    lower_black = np.array([0, 0, 10])
    upper_black = np.array([180, 255, 50])

    # Create HSV black mask
    black_mask_hsv = cv2.inRange(upper_body_hsv, lower_black, upper_black)

    # Grayscale fallback for robustness
    gray_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2GRAY)
    upper_gray = gray_roi[start_y:end_y, crop_margin:width - crop_margin]
    _, gray_mask = cv2.threshold(upper_gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Combine masks
    combined_mask = cv2.bitwise_and(black_mask_hsv, gray_mask)

    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    # Calculate ratio of black pixels
    total_pixels = combined_mask.shape[0] * combined_mask.shape[1]
    black_pixels = cv2.countNonZero(combined_mask)
    black_ratio = black_pixels / total_pixels if total_pixels > 0 else -1
    
    return black_ratio