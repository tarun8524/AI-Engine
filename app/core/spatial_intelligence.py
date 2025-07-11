import cv2 #type: ignore
import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.spatial import distance

class SpatialIntelligence:
    """Spatial analysis utilities for video analytics"""
    
    def __init__(self, max_disappeared: int = 100):
        """Initialize with tracking parameters"""
        self.max_disappeared = max_disappeared
        self.next_object_id = 0
        self.objects = {}  # Tracks current centroids
        self.disappeared = {}  # Tracks disappearance count
    
    @staticmethod
    def calculate_centroid(bbox: List[int]) -> Tuple[int, int]:
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_1)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def register(self, centroid: Tuple[int, int]) -> int:
        """Register new object with centroid"""
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.disappeared[object_id] = 0
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id: int) -> None:
        """Deregister object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
    
    def update_tracker(self, centroids: List[Tuple[int, int]]) -> Dict[int, Tuple[int, int]]:
        """Update tracker with new centroids using Code 1 logic"""
        if len(centroids) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        if len(self.objects) == 0:
            for centroid in centroids:
                self.register(centroid)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            D = distance.cdist(np.array(object_centroids), np.array(centroids))
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                object_id = object_ids[row]
                self.objects[object_id] = centroids[col]
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            if D.shape[0] >= D.shape[1]:
                for row in unused_rows:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_cols:
                    self.register(centroids[col])

        return self.objects
    
    def assign_tracking_ids(self, detections: List[Dict[str, Any]], tracker: Dict, next_id: int) -> Tuple[List[Dict[str, Any]], int]:
        """Assign continuous tracking IDs to detections based on centroid proximity"""
        centroids = [self.calculate_centroid(det['bbox']) for det in detections]
        updated_objects = self.update_tracker(centroids)
        
        updated_detections = []
        used_object_ids = set()
        
        for det, centroid in zip(detections, centroids):
            det_copy = det.copy()
            det_copy['centroid'] = centroid
            
            # Find matching object ID
            assigned = False
            for obj_id, obj_centroid in updated_objects.items():
                if obj_id in used_object_ids:
                    continue
                # Exact match for centroid to ensure consistency
                if centroid == obj_centroid:
                    det_copy['track_id'] = obj_id
                    used_object_ids.add(obj_id)
                    assigned = True
                    break
            
            # If no match found, register new object
            if not assigned:
                new_id = self.register(centroid)
                det_copy['track_id'] = new_id
                used_object_ids.add(new_id)
            
            updated_detections.append(det_copy)
        
        return updated_detections, self.next_object_id
    
    def analyze_detections(self, detections: List[Dict[str, Any]], rule_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze detections and extract spatial intelligence with tracking"""
        spatial_data = {
            'detections': detections,
            'person_detections': [],
            'centroids': [],
            'detection_count': len(detections)
        }
        
        # Filter person detections
        for detection in detections:
            if detection['class_name'] == 'person':
                spatial_data['person_detections'].append(detection)
                spatial_data['centroids'].append(self.calculate_centroid(detection['bbox']))
        
        # Assign tracking IDs based on centroids
        spatial_data['person_detections'], rule_state['next_id'] = self.assign_tracking_ids(
            spatial_data['person_detections'], rule_state['tracker'], rule_state['next_id']
        )
        
        # Update tracker in rule state
        rule_state['tracker'] = {det['track_id']: det['centroid'] for det in spatial_data['person_detections']}
        
        return spatial_data