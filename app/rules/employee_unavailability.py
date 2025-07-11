import cv2 #type: ignore
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio

async def process_employee_unavailability_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Employee Unavailability rule"""
    # MongoDB setup
    collection = get_collection("emp_unavailability")

    # Get FPS from rule state
    fps = rule_state.get('fps', 30)
    print(f"FPS: {fps}")

    # Increment frame count
    rule_state['frame_count'] = rule_state.get('frame_count', 0) + 1
    current_time = rule_state['frame_count'] / fps

    # Initialize state if not present
    if 'current_state' not in rule_state:
        rule_state['current_state'] = "available"
        rule_state['state_start_time'] = 0
        rule_state['pending_state'] = None
        rule_state['pending_state_start_time'] = None
        rule_state['STATE_CHANGE_THRESHOLD'] = 2
        rule_state['DISAPPEARANCE_THRESHOLD'] = 10
        print("Initialized rule state")

    # Log spatial data for debugging
    print(f"Person detections: {spatial_data.get('person_detections', [])}")

    # Check for employees (persons with black shirts)
    employee_present = False
    for detection in spatial_data.get('person_detections', []):
        if detection['confidence'] > 0.75 and detection['class_name'] == 'person':
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
            print(f"Detection: confidence={detection['confidence']}, black_ratio={black_ratio}")
            if black_ratio > 0.2:
                employee_present = True
                # Draw visual indicators
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                cv2.putText(frame, "Employee", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break

    # Determine desired state
    desired_state = "available" if employee_present else "unavailable"
    print(f"Current State: {rule_state['current_state']}, Desired State: {desired_state}, Pending State: {rule_state.get('pending_state')}")

    # Handle 10-second disappearance/appearance logic
    if desired_state != rule_state['current_state']:
        if rule_state['pending_state'] != desired_state:
            rule_state['pending_state'] = desired_state
            rule_state['pending_state_start_time'] = current_time
            print(f"Pending state change to '{desired_state}' started at {current_time:.2f}s")
        else:
            pending_duration = current_time - rule_state['pending_state_start_time']
            if pending_duration >= rule_state['DISAPPEARANCE_THRESHOLD']:
                previous_state_duration = current_time - rule_state['state_start_time']
                if previous_state_duration >= rule_state['STATE_CHANGE_THRESHOLD'] and rule_state['current_state'] == "unavailable":
                    last_state_timestamp = datetime.now()
                    try:
                        collection.insert_one({
                            "timestamp": last_state_timestamp,
                            "camera_name": camera_name,
                            "emp_unavailability": round(previous_state_duration, 2),
                            "session_id": session_id
                        })
                        print(f"DB Update at {current_time:.2f}s: unavailable = {previous_state_duration:.2f}s")
                    except Exception as e:
                        print(f"DB Error at {current_time:.2f}s: {e}")
                # Update state and include pending time in new state duration
                rule_state['current_state'] = desired_state
                # Set state_start_time to pending_state_start_time to include pending period
                rule_state['state_start_time'] = rule_state['pending_state_start_time']
                rule_state['pending_state'] = None
                rule_state['pending_state_start_time'] = None
                print(f"State changed to {rule_state['current_state']} at {current_time:.2f}s, starting from {rule_state['state_start_time']:.2f}s")
    else:
        if rule_state['pending_state'] is not None:
            print(f"Pending state change to '{rule_state['pending_state']}' cancelled at {current_time:.2f}s")
            rule_state['pending_state'] = None
            rule_state['pending_state_start_time'] = None

    # Calculate current duration for display
    current_duration = current_time - rule_state['state_start_time']

    # Create status text
    status_text = f"State: {rule_state['current_state']} | Duration: {int(current_duration)}s"
    if rule_state['pending_state'] is not None:
        pending_duration = current_time - rule_state['pending_state_start_time']
        remaining_time = rule_state['DISAPPEARANCE_THRESHOLD'] - pending_duration
        status_text += f" | Pending: {rule_state['pending_state']} ({remaining_time:.1f}s left)"
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Debug information
    debug_info = f"Frame {rule_state['frame_count']}: Employee Present = {employee_present}, Current State = {rule_state['current_state']}, Duration = {current_duration:.1f}s"
    if rule_state['pending_state'] is not None:
        pending_duration = current_time - rule_state['pending_state_start_time']
        debug_info += f", Pending = {rule_state['pending_state']} ({pending_duration:.1f}s/{rule_state['DISAPPEARANCE_THRESHOLD']}s)"
    print(debug_info)