import cv2 #type: ignore
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.inside_roi import is_inside_zone

def detect_billing(printer_crop, white_area_threshold=450):
    gray = cv2.cvtColor(printer_crop, cv2.COLOR_BGR2GRAY)
    _, thresh_simple = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    white_area = np.sum(thresh_simple == 255)
    return white_area > white_area_threshold, white_area

def roi_to_polygon(roi):
    """Convert ROI tuple (x, y, w, h) to polygon points format for cv2.pointPolygonTest"""
    x, y, w, h = roi
    # Create polygon points in the format expected by cv2.pointPolygonTest: (n, 1, 2)
    polygon = np.array([
        [[x, y]],
        [[x + w, y]],
        [[x + w, y + h]],
        [[x, y + h]]
    ], dtype=np.float32)
    return polygon

def detect_customer_in_roi(customer_roi, spatial_data, debug=True):
    """
    Detect customer presence in ROI using centroid or bottom-center point.
    Returns True if any person is inside the ROI polygon.
    """
    polygon = roi_to_polygon(customer_roi).reshape(-1, 2).astype(np.int32)  # cv2 polygon format

    for i, det in enumerate(spatial_data.get('person_detections', [])):
        if det.get('confidence', 0) < 0.5:
            continue
        
        x1, y1, x2, y2 = det['bbox']
        centroid_x, centroid_y = det['centroid']
        bottom_center = ((x1 + x2) // 2, y2)

        # Check both centroid and bottom center
        if is_inside_zone(centroid_x, centroid_y, polygon) or is_inside_zone(*bottom_center, polygon):
            if debug:
                print(f"Person {i} inside ROI via centroid or bottom center.")
            return True
        else:
            if debug:
                print(f"Person {i} NOT in ROI. Centroid: ({centroid_x}, {centroid_y}), Bottom: {bottom_center}")
    
    return False


async def process_billing_alerts_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Billing Alerts rule to detect billing events and customer presence."""
    # Initialize state if not present
    if 'continuous_counter' not in rule_state:
        rule_state['continuous_counter'] = 0
        rule_state['paper_absent_counter'] = 0
        rule_state['billing_done'] = False
        rule_state['last_billing_state'] = False
        rule_state['billing_count'] = 0
        rule_state['last_alert_reason'] = None
        rule_state['customer_no_bill_start'] = None
        rule_state['bill_no_customer_start'] = None
        rule_state['last_count_timestamp'] = datetime.now()
        rule_state['fps'] = 30
        rule_state['frame_count'] = 0
    
    # Ensure last_count_timestamp is never None
    if rule_state['last_count_timestamp'] is None:
        rule_state['last_count_timestamp'] = datetime.now()

    # Constants
    CONTINUOUS_REQUIRED = 5
    RESET_REQUIRED = 25
    CUSTOMER_NO_BILL_THRESHOLD = timedelta(seconds=1)
    BILL_NO_CUSTOMER_THRESHOLD = timedelta(seconds=1)
    BILLING_COUNT_INTERVAL = timedelta(seconds=60)

    # MongoDB setup
    alert_collection = get_collection("billing_alerts")
    count_collection = get_collection("billing_count")

    # Standard ROIs
    printer_roi = (880, 803, 187, 29)  # (x, y, w, h)
    customer_roi = (122, 142, 548, 690)  # (x, y, w, h)
    
    # Convert ROIs to polygon format for is_inside_zone function
    customer_polygon = roi_to_polygon(customer_roi)

    # Draw ROIs
    x, y, w, h = printer_roi
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cx, cy, cw, ch = customer_roi
    cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (255, 0, 0), 2)

    # Increment frame count
    rule_state['frame_count'] += 1

    # Detect billing in printer ROI
    printer_crop = frame[y:y+h, x:x+w]
    paper_detected, white_area = detect_billing(printer_crop)

    # Detect customer in customer ROI using the improved logic
    customer_present = detect_customer_in_roi(customer_roi, spatial_data, debug=False)  # Set debug=True to see detection info
    
    # Fallback: Simple overlap check if the above doesn't work
    if not customer_present:
        cx, cy, cw, ch = customer_roi
        for det in spatial_data.get('person_detections', []):
            if det['confidence'] >= 0.3:  # Lower confidence threshold for fallback
                x1, y1, x2, y2 = det['bbox']
                centroid_x, centroid_y = det['centroid']
                
                # Simple overlap check
                if (x1 < cx + cw and x2 > cx and y1 < cy + ch and y2 > cy):
                    customer_present = True
                    break

    # Update billing status
    if paper_detected:
        rule_state['continuous_counter'] += 1
        rule_state['paper_absent_counter'] = 0
    else:
        rule_state['continuous_counter'] = 0
        rule_state['paper_absent_counter'] += 1

    if rule_state['continuous_counter'] >= CONTINUOUS_REQUIRED:
        rule_state['billing_done'] = True
    if rule_state['paper_absent_counter'] >= RESET_REQUIRED:
        rule_state['billing_done'] = False

    # Track billing count on transition
    if rule_state['billing_done'] and not rule_state['last_billing_state']:
        rule_state['billing_count'] += 1
        print(f"Billing event detected! Billing count: {rule_state['billing_count']}")
    rule_state['last_billing_state'] = rule_state['billing_done']

    # Save billing count to DB every 60 seconds
    current_time = datetime.now()
    if current_time - rule_state['last_count_timestamp'] >= BILLING_COUNT_INTERVAL:
        if rule_state['billing_count'] >= 1:
            data = {
                "timestamp": current_time,
                "camera_name": camera_name,
                "billing_count": rule_state['billing_count']
            }
            try:
                count_collection.insert_one(data)
                print(f"Inserted billing count to DB: {rule_state['billing_count']} at {current_time}")
            except Exception as e:
                print(f"Error inserting billing count to DB: {e}")
            rule_state['billing_count'] = 0
        rule_state['last_count_timestamp'] = current_time

    # Alert logic
    current_alert = False
    current_reason = ""

    if customer_present and not rule_state['billing_done']:
        if rule_state['customer_no_bill_start'] is None:
            rule_state['customer_no_bill_start'] = datetime.now()
        elif datetime.now() - rule_state['customer_no_bill_start'] >= CUSTOMER_NO_BILL_THRESHOLD:
            current_alert = True
            current_reason = "Customer Present but No Bill Printed"
    else:
        rule_state['customer_no_bill_start'] = None

    if rule_state['billing_done'] and not customer_present:
        if rule_state['bill_no_customer_start'] is None:
            rule_state['bill_no_customer_start'] = datetime.now()
        elif datetime.now() - rule_state['bill_no_customer_start'] >= BILL_NO_CUSTOMER_THRESHOLD:
            current_alert = True
            current_reason = "Bill Printed but Customer Not Present"
    else:
        rule_state['bill_no_customer_start'] = None

    if not current_alert:
        rule_state['last_alert_reason'] = None

    if current_alert and current_reason != rule_state['last_alert_reason']:
        data = {
            "timestamp": datetime.now(),
            "camera_name": camera_name,
            "alerts": True,
            "reason": current_reason
        }
        try:
            alert_collection.insert_one(data)
            rule_state['last_alert_reason'] = current_reason
            print(f"Inserted alert to DB: {current_reason} at {datetime.now()}")
        except Exception as e:
            print(f"Error inserting alert to DB: {e}")

    # Visualizations
    frame_width = frame.shape[1]
    if customer_present:
        cv2.putText(frame, "Customer Present", (frame_width - 350, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    else:
        cv2.putText(frame, "Customer Not Present", (frame_width - 450, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    if rule_state['billing_done']:
        cv2.putText(frame, "Billing Completed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
    elif paper_detected:
        cv2.putText(frame, "Detecting Bill...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if current_alert:
        cv2.putText(frame, f"ALERT: {current_reason.upper()}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, f"Billing Count: {rule_state['billing_count']}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)