import cv2 #type: ignore
import numpy as np
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio

def is_customer_present(frame, detections, zone_points):
    """Check if a customer (person without black shirt) is present and return customer detections"""
    if len(detections) == 0:
        return False, []

    customer_detections = []

    for det in detections:
        try:
            x1, y1, x2, y2 = map(int, det['bbox'])
            black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
            if black_ratio < 0.2 and black_ratio >= 0:
                customer_detections.append(det)
        except Exception as e:
            print(f"Black ratio check error: {e}")
            continue

    return len(customer_detections) > 0, customer_detections


async def process_occupancy_monitoring_rule(frame: np.ndarray, spatial_data: Dict[str, Any],
                                            camera_name: str, session_id: str,
                                            mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Insert customer count every 20 seconds if new customer IDs appear ≥10 times in last 20 frames"""

    collection = get_collection("occupancy_monitoring")

    # Initialize rule state
    if 'id_history_buffer' not in rule_state:
        rule_state['id_history_buffer'] = deque(maxlen=20)
    if 'inserted_ids' not in rule_state:
        rule_state['inserted_ids'] = set()
    if 'last_insertion_time' not in rule_state:
        rule_state['last_insertion_time'] = None

    all_detections = spatial_data.get('person_detections', [])
    _, customer_detections = is_customer_present(frame, all_detections, spatial_data.get("zone_points", []))

    # Track only customer IDs
    current_ids = {det['track_id'] for det in customer_detections if 'track_id' in det}
    rule_state['id_history_buffer'].append(current_ids)

    # Draw customer detections
    for det in customer_detections:
        try:
            x1, y1, x2, y2 = map(int, det['bbox'])
            cx, cy = map(int, det['centroid'])
            track_id = det['track_id']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)
        except Exception as e:
            print(f"Draw error: {e}")

    # Count frequency of each ID over last 20 frames
    id_freq = defaultdict(int)
    for id_set in rule_state['id_history_buffer']:
        for tid in id_set:
            id_freq[tid] += 1

    # Time check
    now = datetime.now()
    time_elapsed = (
        rule_state['last_insertion_time'] is None or
        (now - rule_state['last_insertion_time']).total_seconds() >= 20
    )

    if time_elapsed:
        # Get new stable customer IDs
        new_stable_ids = [
            tid for tid, freq in id_freq.items()
            if freq >= 10 and tid not in rule_state['inserted_ids']
        ]

        if new_stable_ids:
            try:
                collection.insert_one({
                    "timestamp": now,
                    "camera_name": camera_name,
                    "person_count": len(new_stable_ids)
                })
                rule_state['inserted_ids'].update(new_stable_ids)
                rule_state['last_insertion_time'] = now
                print(f"[{camera_name}] Inserted customer count: {len(new_stable_ids)} at {now}")
            except Exception as e:
                print(f"DB Insert error: {e}")
