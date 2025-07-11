import cv2 #type: ignore
import numpy as np
import os
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.inside_roi import is_inside_zone

async def process_intrusion_rule(frame: np.ndarray, spatial_data: Dict[str, Any], 
                                 camera_name: str, session_id: str, 
                                 mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process intrusion detection rule using track IDs"""

    # Define ROI
    roi_data = [
        [400, 300],
        [750, 300],
        [750, 800],
        [400, 800]
    ]
    roi_coordinates_int = np.array(roi_data, dtype=np.int32)
    cv2.polylines(frame, [roi_coordinates_int], isClosed=True, color=(0, 255, 255), thickness=2)

    # MongoDB collection
    collection = get_collection("intrusion")

    # Folder for saving intrusion frames
    frames_folder = "frames/intrusion_frames"
    os.makedirs(frames_folder, exist_ok=True)

    # Initialize nested rule state
    intrusion_state = rule_state.setdefault('Intrusion', {})
    if 'intruded_ids' not in intrusion_state:
        intrusion_state['intruded_ids'] = set()

    intrusion_detected = False
    current_intrusion_ids = set()

    for detection in spatial_data.get('person_detections', []):
        confidence = detection.get('confidence', 0)
        if confidence > 0.7:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            bottom_center = ((x1 + x2) // 2, y2)
            track_id = detection.get('track_id')

            # Check ROI
            if is_inside_zone(cx, cy, roi_coordinates_int) or is_inside_zone(*bottom_center, roi_coordinates_int):
                intrusion_detected = True
                current_intrusion_ids.add(track_id)
                color = (0, 0, 255)
                label = f"Intrusion ID:{track_id}"
            else:
                color = (0, 255, 0)
                label = f"Person ID:{track_id}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Determine new intrusions
    new_intrusions = current_intrusion_ids - intrusion_state['intruded_ids']

    if new_intrusions:
        timestamp = datetime.now()
        for track_id in new_intrusions:
            data = {
                "timestamp": timestamp,
                "intrusion": True,
                "track_id": track_id,
                "session_id": session_id,
                "camera_name": camera_name
            }
            try:
                collection.insert_one(data)
            except Exception as e:
                print(f"Error inserting to DB: {e}")

        # Save the frame once
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        frame_filename = f"intrusion_{session_id}_{camera_name}_{timestamp_str}.jpg"
        frame_path = os.path.join(frames_folder, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Update seen intruder IDs
        intrusion_state['intruded_ids'].update(new_intrusions)

    # Update global flag (optional use)
    intrusion_state['prev_intrusion_detected'] = intrusion_detected
