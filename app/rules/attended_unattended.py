import cv2 #type: ignore
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio
from app.rules.utils.inside_roi import is_inside_zone

async def process_attended_unattended_time_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Attended/Unattended Time rule"""
    zone = np.array([
        [400, 300],
        [750, 300],
        [750, 800],
        [400, 800]
    ], dtype=np.int32)

    # Draw ROI
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

    collection = get_collection("attended_unattended")
    fps = rule_state.get('fps', 30)
    rule_state['frame_count'] = rule_state.get('frame_count', 0) + 1
    current_time = rule_state['frame_count'] / fps

    if 'current_state' not in rule_state:
        rule_state.update({
            'current_state': "unattended",
            'state_start_time': 0,
            'pending_state': None,
            'pending_state_start_time': None,
            'STATE_CHANGE_THRESHOLD': 2,
            'DISAPPEARANCE_THRESHOLD': 10
        })

    employee_in_roi = False
    for detection in spatial_data.get('person_detections', []):
        if detection['confidence'] > 0.75 and detection['class_name'] == 'person':
            x1, y1, x2, y2 = detection['bbox']
            black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
            if black_ratio > 0.2:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                bottom_center_x = (x1 + x2) // 2
                bottom_center_y = y2

                centroid_inside = is_inside_zone(cx, cy, zone)
                bottom_center_inside = is_inside_zone(bottom_center_x, bottom_center_y, zone)

                if centroid_inside or bottom_center_inside:
                    employee_in_roi = True
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
                    cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, "Employee", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    break

    desired_state = "attended" if employee_in_roi else "unattended"

    if desired_state != rule_state['current_state']:
        if rule_state['pending_state'] != desired_state:
            rule_state['pending_state'] = desired_state
            rule_state['pending_state_start_time'] = current_time
            print(f"Pending state change to '{desired_state}' started at {current_time:.2f}s")
        else:
            pending_duration = current_time - rule_state['pending_state_start_time']
            if pending_duration >= rule_state['DISAPPEARANCE_THRESHOLD']:
                previous_state_duration = current_time - rule_state['state_start_time']
                if previous_state_duration >= rule_state['STATE_CHANGE_THRESHOLD']:
                    try:
                        collection.insert_one({
                            "timestamp": datetime.now(),
                            "camera_name": camera_name,
                            "type": rule_state['current_state'],
                            "time_duration": round(previous_state_duration, 2),
                            "session_id": session_id
                        })
                        print(f"DB Update at {current_time:.2f}s: {rule_state['current_state']} = {previous_state_duration:.2f}s")
                    except Exception as e:
                        print(f"DB Error at {current_time:.2f}s: {e}")
                rule_state['current_state'] = desired_state
                rule_state['state_start_time'] = current_time
                rule_state['pending_state'] = None
                rule_state['pending_state_start_time'] = None
                print(f"State changed to {rule_state['current_state']} at {current_time:.2f}s after 10 seconds")
    else:
        if rule_state['pending_state'] is not None:
            print(f"Pending state change to '{rule_state['pending_state']}' cancelled at {current_time:.2f}s")
            rule_state['pending_state'] = None
            rule_state['pending_state_start_time'] = None

    current_duration = current_time - rule_state['state_start_time']
    status_text = f"State: {rule_state['current_state']} | Duration: {int(current_duration)}s"
    if rule_state['pending_state'] is not None:
        pending_duration = current_time - rule_state['pending_state_start_time']
        remaining_time = rule_state['DISAPPEARANCE_THRESHOLD'] - pending_duration
        status_text += f" | Pending: {rule_state['pending_state']} ({remaining_time:.1f}s left)"
    cv2.putText(frame, status_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    debug_info = f"Frame {rule_state['frame_count']}: Employee in ROI = {employee_in_roi}, Current State = {rule_state['current_state']}, Duration = {current_duration:.1f}s"
    if rule_state['pending_state'] is not None:
        pending_duration = current_time - rule_state['pending_state_start_time']
        debug_info += f", Pending = {rule_state['pending_state']} ({pending_duration:.1f}s/{rule_state['DISAPPEARANCE_THRESHOLD']}s)"
    print(debug_info)
