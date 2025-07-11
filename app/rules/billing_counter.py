import cv2 #type: ignore
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.inside_roi import is_inside_zone

async def process_billing_counter_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Billing Counter rule to count people using centroid or bottom-center inside ROI."""
    
    # Initialize state if not present
    if 'count_history' not in rule_state:
        rule_state['count_history'] = deque(maxlen=20)
        rule_state['first_timestamp'] = None
        rule_state['last_inserted_count'] = None
        rule_state['frame_count'] = 0
        rule_state['fps'] = 30

    # Define ROI
    zone = np.array([
        [400, 300],
        [750, 300],
        [750, 800],
        [400, 800]
    ], dtype=np.int32)

    # Draw ROI on the frame
    cv2.polylines(frame, [zone], isClosed=True, color=(0, 0, 255), thickness=2)

    # MongoDB setup
    collection = get_collection("billing_counter")

    rule_state['frame_count'] += 1
    person_count = 0

    for det in spatial_data.get('person_detections', []):
        x1, y1, x2, y2 = det['bbox']
        cx, cy = det['centroid']
        bottom_center = ((x1 + x2) // 2, y2)

        # Check if either centroid or bottom center is inside ROI
        if (is_inside_zone(cx, cy, zone) or is_inside_zone(*bottom_center, zone)) and det['confidence'] >= 0.6:
            person_count += 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
            cv2.circle(frame, bottom_center, 5, (0, 0, 255), -1)  # Optional: visualize bottom-center

    # Update count history
    rule_state['count_history'].append(person_count)

    # Track first timestamp for current count
    if not rule_state['count_history'] or (len(rule_state['count_history']) >= 2 and
                                          rule_state['count_history'][-1] != rule_state['count_history'][-2]):
        rule_state['first_timestamp'] = datetime.now()

    # Check for consistent count (15/20 frames)
    if len(rule_state['count_history']) >= 20:
        current_count = rule_state['count_history'][-1]
        count_frequency = sum(1 for count in rule_state['count_history'] if count == current_count)

        if count_frequency >= 15 and current_count != rule_state['last_inserted_count'] and current_count > 0:
            timestamp = datetime.now()
            data = {
                "timestamp": timestamp,
                "camera_name": camera_name,
                "person_count": current_count
            }
            try:
                collection.insert_one(data)
                rule_state['last_inserted_count'] = current_count
                print(f"Inserted person count to DB: {current_count} at {rule_state['first_timestamp']}")
            except Exception as e:
                print(f"Error inserting to DB: {e}")

    # Display count on frame
    cv2.putText(frame, f'People in Zone: {person_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
