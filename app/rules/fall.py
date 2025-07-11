import cv2 #type: ignore
import numpy as np
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Any, Tuple
from app.db.database import get_collection
from app.rules.utils.inside_roi import is_inside_zone
import os
import logging

logger = logging.getLogger(__name__)


class CentroidTracker:
    def __init__(self, max_disappeared=50, max_distance=100):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, orientation):
        self.objects[self.next_object_id] = {
            'centroid': centroid,
            'bbox': bbox,
            'orientation_history': [orientation],
            'fall_confirmed_frame': None,
            'fall_state': 'normal',
            'velocity': (0, 0),
            'predicted_centroid': centroid
        }
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, detections, frame_count):
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.array([det['center'] for det in detections])

        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection['center'], detection['bbox'], detection['orientation'])
        else:
            object_centroids = []
            object_ids = list(self.objects.keys())
            
            for object_id in object_ids:
                obj = self.objects[object_id]
                predicted_pos = (
                    obj['centroid'][0] + obj['velocity'][0],
                    obj['centroid'][1] + obj['velocity'][1]
                )
                self.objects[object_id]['predicted_centroid'] = predicted_pos
                object_centroids.append(predicted_pos)

            object_centroids = np.array(object_centroids)
            D = np.linalg.norm(object_centroids[:, np.newaxis] - input_centroids, axis=2)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_row_indices = set()
            used_col_indices = set()

            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue

                if D[row, col] > self.max_distance:
                    continue

                object_id = object_ids[row]
                detection = detections[col]
                
                old_centroid = self.objects[object_id]['centroid']
                new_centroid = detection['center']
                velocity = (new_centroid[0] - old_centroid[0], new_centroid[1] - old_centroid[1])
                
                self.objects[object_id]['centroid'] = new_centroid
                self.objects[object_id]['bbox'] = detection['bbox']
                self.objects[object_id]['velocity'] = velocity
                
                self.objects[object_id]['orientation_history'].append(detection['orientation'])
                if len(self.objects[object_id]['orientation_history']) > 7:
                    self.objects[object_id]['orientation_history'].pop(0)
                
                self.disappeared[object_id] = 0

                used_row_indices.add(row)
                used_col_indices.add(col)

            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)

            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    detection = detections[col]
                    self.register(detection['center'], detection['bbox'], detection['orientation'])

        return self.objects

async def process_fall_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Fall detection rule to identify people falling based on orientation and velocity."""
    try:
        # Ensure frames_folder exists
        frames_folder = "frames/fall_frames"

        # Constants
        ASPECT_RATIO_THRESHOLD = 1.2
        VERTICAL_THRESHOLD = 0.85
        FALL_SPEED_THRESHOLD = 12
        Y_VELOCITY_THRESHOLD = 2
        CONFIRMATION_FRAMES = 6
        MIN_PERSON_SIZE = 800

        # MongoDB setup
        collection = get_collection("fall")

        # Increment frame count
        rule_state['frame_count'] += 1

        # Ensure tracker exists
        if 'tracker' not in rule_state or rule_state['tracker'] is None:
            logger.error(f"Tracker is None for camera {camera_name}. Initializing new CentroidTracker.")
            rule_state['tracker'] = CentroidTracker(max_disappeared=20, max_distance=80)

        def get_orientation(w, h):
            aspect_ratio = w / h
            if aspect_ratio > ASPECT_RATIO_THRESHOLD:
                return "horizontal"
            elif aspect_ratio < (1 / VERTICAL_THRESHOLD):
                return "vertical"
            else:
                return "square"

        def detect_sudden_transition(orientation_history):
            if len(orientation_history) < 4:
                return False
            recent_horizontal = orientation_history[-2:].count("horizontal") >= 1
            previous_vertical = orientation_history[:-2].count("vertical") >= 2
            return recent_horizontal and previous_vertical

        def analyze_fall_pattern(orientation_history, velocity):
            if len(orientation_history) < 3:
                return False, "insufficient_data"
            current_orientation = orientation_history[-1]
            sudden_transition = detect_sudden_transition(orientation_history)
            speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
            y_velocity = velocity[1]
            is_horizontal_now = current_orientation == "horizontal"
            sufficient_speed = speed > FALL_SPEED_THRESHOLD
            downward_movement = y_velocity > Y_VELOCITY_THRESHOLD
            if sudden_transition and is_horizontal_now and sufficient_speed and downward_movement:
                return True, "fall_detected"
            elif is_horizontal_now and speed < 3:
                return True, "lying_still"
            else:
                return False, "normal"

        # Process detections from spatial_data
        current_detections = []
        for det in spatial_data.get('person_detections', []):
            x1, y1, x2, y2 = det['bbox']
            w, h = x2 - x1, y2 - y1
            area = w * h
            if area < MIN_PERSON_SIZE or det['confidence'] < 0.6:
                continue
            cx, cy = det['centroid']
            current_detections.append({
                'bbox': (x1, y1, x2, y2),
                'center': (cx, cy),
                'width': w,
                'height': h,
                'area': area,
                'orientation': get_orientation(w, h)
            })

        # Update tracker
        try:
            objects = rule_state['tracker'].update(current_detections, rule_state['frame_count'])
        except Exception as e:
            logger.error(f"Error updating tracker for camera {camera_name}: {e}")
            rule_state['tracker'] = CentroidTracker(max_disappeared=20, max_distance=80)
            objects = rule_state['tracker'].update(current_detections, rule_state['frame_count'])

        fall_detected = False
        for object_id, obj_data in objects.items():
            x1, y1, x2, y2 = obj_data['bbox']
            is_fall, fall_reason = analyze_fall_pattern(obj_data['orientation_history'], obj_data['velocity'])
            
            if is_fall and fall_reason == "fall_detected":
                if obj_data['fall_state'] == 'normal':
                    obj_data['fall_confirmed_frame'] = rule_state['frame_count']
                    obj_data['fall_state'] = 'potential_fall'
            
            label = None
            color = (0, 255, 0)
            
            if obj_data['fall_state'] == 'potential_fall':
                if (rule_state['frame_count'] - obj_data['fall_confirmed_frame'] >= CONFIRMATION_FRAMES and
                    obj_data['orientation_history'][-1] == "horizontal"):
                    obj_data['fall_state'] = 'fallen'
                    fall_detected = True
                    label = "ALERT: FALL/SLIP DETECTED"
                    color = (0, 0, 255)
                elif obj_data['orientation_history'][-1] == "vertical":
                    obj_data['fall_state'] = 'normal'
                    obj_data['fall_confirmed_frame'] = None
                    
            elif obj_data['fall_state'] == 'fallen':
                current_orientation = obj_data['orientation_history'][-1]
                speed = np.sqrt(obj_data['velocity'][0]**2 + obj_data['velocity'][1]**2)
                if current_orientation == "vertical" and speed > 5:
                    obj_data['fall_state'] = 'normal'
                    obj_data['fall_confirmed_frame'] = None
                else:
                    fall_detected = True
                    label = "ALERT: FALL/SLIP DETECTED"
                    color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            if label:
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save to MongoDB if fall detected and not previously detected
        if fall_detected and not rule_state['prev_fall_detected']:
            timestamp = datetime.now()
            data = {
                "timestamp": timestamp,
                "Alert": fall_detected,
                "camera_name": camera_name
            }
            try:
                collection.insert_one(data)
                # Save frame to local folder
                timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                frame_filename = f"fall_{session_id}_{camera_name}_{timestamp_str}.jpg"
                frame_path = os.path.join(frames_folder, frame_filename)
                cv2.imwrite(frame_path, frame)
                logger.info(f"Saved fall frame to {frame_path}")
            except Exception as e:
                logger.error(f"Error inserting to DB or saving frame for camera {camera_name}: {e}")

        rule_state['prev_fall_detected'] = fall_detected

    except Exception as e:
        logger.error(f"Error in process_fall_rule for camera {camera_name}: {e}")
        raise