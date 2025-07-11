# app/rules/dwell_time.py
import cv2 #type: ignore
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio
from app.rules.utils.inside_roi import is_inside_zone


def is_customer_present(frame, detections, zone_points):
    """Check if a customer (person without black shirt) is present within the ROI and return customer detections"""
    if len(detections) == 0:
        return False, []
    
    customer_detections = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.75:
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            bottom_center_x = (x1 + x2) / 2
            bottom_center_y = y2

            # Check if centroid or bottom-center is inside ROI
            if is_inside_zone(cx, cy, zone_points) or is_inside_zone(bottom_center_x, bottom_center_y, zone_points):
                black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
                if black_ratio < 0.2 and black_ratio >= 0:
                    customer_detections.append(det)
    
    return len(customer_detections) > 0, customer_detections


async def process_dwell_time_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Dwell Time rule to track customer dwell time within ROI."""
    # Define the ROI points
    roi_points = np.array([
        [400, 300],
        [750, 300],
        [750, 800],
        [400, 800]
    ], dtype=np.int32).reshape((-1, 1, 2))

    # Draw the ROI on the frame
    cv2.polylines(frame, [roi_points], isClosed=True, color=(255, 255, 0), thickness=2)

    # MongoDB setup
    collection = get_collection("customer_dwell_time")

    # Get FPS from rule state
    fps = rule_state.get('fps', 30)

    # Increment frame count and calculate current time
    rule_state['frame_count'] = rule_state.get('frame_count', 0) + 1
    current_time = rule_state['frame_count'] / fps

    # Initialize state if not present
    if 'customer_present' not in rule_state:
        rule_state['customer_present'] = False
        rule_state['dwell_start_time'] = 0
        rule_state['accumulated_dwell_time'] = 0
        rule_state['last_customer_time'] = 0
        rule_state['no_customer_frames'] = 0
        rule_state['MIN_NO_CUSTOMER_FRAMES'] = 25

    # Convert spatial_data detections to format expected by is_customer_present
    formatted_detections = [
        (
            det['bbox'][0],  # x1
            det['bbox'][1],  # y1
            det['bbox'][2],  # x2
            det['bbox'][3],  # y2
            det['confidence'],  # score
            0 if det['class_name'] == 'person' else -1  # class_id (assuming person is class 0)
        )
        for det in spatial_data.get('detections', [])
    ]

    # Check if customer is present within ROI
    customer_detected, customer_detections = is_customer_present(frame, formatted_detections, roi_points)

    # Update dwell time tracking
    if customer_detected:
        if not rule_state['customer_present']:
            if rule_state['no_customer_frames'] < rule_state['MIN_NO_CUSTOMER_FRAMES']:
                pass  # Continue previous session
            else:
                rule_state['dwell_start_time'] = current_time
                rule_state['accumulated_dwell_time'] = 0
            rule_state['customer_present'] = True
        rule_state['no_customer_frames'] = 0
        rule_state['last_customer_time'] = current_time
    else:
        if rule_state['customer_present']:
            rule_state['no_customer_frames'] += 1
            rule_state['accumulated_dwell_time'] = rule_state['last_customer_time'] - rule_state['dwell_start_time']
            if rule_state['no_customer_frames'] >= rule_state['MIN_NO_CUSTOMER_FRAMES']:
                if rule_state['accumulated_dwell_time'] > 2:
                    try:
                        collection.insert_one({
                            "timestamp": datetime.now(),
                            "camera_name": camera_name,
                            "customer_dwell_time": round(rule_state['accumulated_dwell_time'], 2),
                            "session_id": session_id
                        })
                        print(f"DB Update at {current_time:.2f}s: Dwell Time = {rule_state['accumulated_dwell_time']:.2f}s")
                    except Exception as e:
                        print(f"DB Error at {current_time:.2f}s: {e}")
                rule_state['customer_present'] = False
                rule_state['no_customer_frames'] = 0
                rule_state['accumulated_dwell_time'] = 0

    # Draw bounding boxes and text
    for det in customer_detections:
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.70:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"Customer: {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display dwell time
    if rule_state['customer_present']:
        current_dwell = rule_state['accumulated_dwell_time'] + (current_time - rule_state['last_customer_time'] if customer_detected else 0)
        alert_text = f"Customer Dwell Time: {int(current_dwell)}s"
        cv2.putText(frame, alert_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        alert_text = "No Customer Present"
        cv2.putText(frame, alert_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2, cv2.LINE_AA)

    # Debug information
    print(f"Frame {rule_state['frame_count']}: Customer Present = {customer_detected}, "
          f"Dwell Time = {rule_state['accumulated_dwell_time']:.2f}s, No Customer Frames = {rule_state['no_customer_frames']}")