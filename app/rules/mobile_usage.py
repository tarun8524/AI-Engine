import cv2 #type: ignore
import numpy as np
import math
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio

def calculate_centroid(box):
    """Calculate the centroid of a bounding box (x1, y1, x2, y2)."""
    x_center = (box[0] + box[2]) / 2
    y_center = (box[1] + box[3]) / 2
    return x_center, y_center

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

async def process_mobile_usage_rule(frame: np.ndarray, spatial_data: Dict[str, Any],
                                   camera_name: str, session_id: str,
                                   mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process mobile usage rule to detect employee phone usage"""
    
    # MongoDB setup
    collection = get_collection("mobile_usage")
    
    # Initialize rule state if not exists
    if 'usage_buffer' not in rule_state:
        rule_state['usage_buffer'] = []
    if 'usage_start_time' not in rule_state:
        rule_state['usage_start_time'] = None
    if 'last_usage_time' not in rule_state:
        rule_state['last_usage_time'] = None
    if 'is_usage_confirmed' not in rule_state:
        rule_state['is_usage_confirmed'] = False
    if 'confirmation_frames' not in rule_state:
        rule_state['confirmation_frames'] = 20
    if 'distance_threshold' not in rule_state:
        rule_state['distance_threshold'] = 400
    
    # Check for employee and phone proximity
    phone_usage = False
    employee_detections = []
    phone_detections = []
    
    for detection in spatial_data['detections']:  # Use all detections, not just person_detections
        x1, y1, x2, y2 = map(int, detection['bbox'])
        score = detection.get('score', 1.0)
        class_name = detection['class_name']
        if class_name == 'person' and score > 0.6:
            black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
            if black_ratio > 0.2:
                employee_detections.append(detection)
        elif class_name == 'cell phone' and score > 0.6:
            phone_detections.append(detection)
    
    for emp_det in employee_detections:
        emp_centroid = emp_det['centroid']
        for phone_det in phone_detections:
            phone_centroid = phone_det['centroid']
            distance = calculate_distance(emp_centroid, phone_centroid)
            if distance < rule_state['distance_threshold']:
                phone_usage = True
                # Draw bounding boxes
                x1, y1, x2, y2 = map(int, emp_det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                x1, y1, x2, y2 = map(int, phone_det['bbox'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Employee Phone Usage", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                break
        if phone_usage:
            break
    
    # Append current frame's phone usage status to buffer
    current_time = datetime.now()
    rule_state['usage_buffer'].append((phone_usage, current_time))
    if len(rule_state['usage_buffer']) > rule_state['confirmation_frames'] * 2:
        rule_state['usage_buffer'].pop(0)  # Keep buffer size to 2 * confirmation_frames
    
    # Check for confirmed phone usage
    if not rule_state['is_usage_confirmed'] and len(rule_state['usage_buffer']) >= rule_state['confirmation_frames'] * 2:
        first_window_usage = any(usage for usage, _ in rule_state['usage_buffer'][:rule_state['confirmation_frames']])
        second_window_usage = any(usage for usage, _ in rule_state['usage_buffer'][rule_state['confirmation_frames']:])
        if first_window_usage and second_window_usage:
            for usage, time in rule_state['usage_buffer'][:rule_state['confirmation_frames']]:
                if usage:
                    rule_state['usage_start_time'] = time
                    break
            rule_state['is_usage_confirmed'] = True
            rule_state['last_usage_time'] = current_time
            print(f"Phone usage confirmed at {rule_state['usage_start_time']}")
    
    # Update end time if usage is ongoing
    if rule_state['is_usage_confirmed'] and phone_usage:
        rule_state['last_usage_time'] = current_time
    
    # End usage event if no usage in last confirmation_frames
    if rule_state['is_usage_confirmed'] and len(rule_state['usage_buffer']) >= rule_state['confirmation_frames'] * 2:
        recent_usage = any(usage for usage, _ in rule_state['usage_buffer'][-rule_state['confirmation_frames']:])
        if not recent_usage:
            data = {
                "session_id": session_id,
                "camera_name": camera_name,
                "timestamp": datetime.now(),
                "duration_seconds": round((rule_state['last_usage_time'] - rule_state['usage_start_time']).total_seconds(), 2)
            }
            try:
                collection.insert_one(data)
                print(f"Saved phone usage event: {data}")
            except Exception as e:
                print(f"Error inserting to DB: {e}")
            # Reset state
            rule_state['is_usage_confirmed'] = False
            rule_state['usage_start_time'] = None
            rule_state['last_usage_time'] = None
            rule_state['usage_buffer'] = []
    
    # Update rule state
    rule_state['prev_phone_usage'] = phone_usage