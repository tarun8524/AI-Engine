import asyncio
import threading
from typing import Optional, List, Dict, Any
from ultralytics import YOLO #type: ignore
import cv2 #type: ignore
import numpy as np
import logging

logger = logging.getLogger(__name__)

class YOLOManager:
    """Singleton YOLO model manager for shared inference across all cameras with support for multiple models"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.models: Dict[str, YOLO] = {}
            self.model_lock = threading.Lock()
            self.initialized = False
    
    async def initialize_model(self, model_path: str = "yolov8m.pt"):
        """Initialize a YOLO model for a specific model path"""
        try:
            with self.model_lock:
                if model_path not in self.models:
                    self.models[model_path] = YOLO(model_path)
                    logger.info(f"YOLO model loaded: {model_path}")
                if not self.initialized:
                    self.initialized = True
        except Exception as e:
            logger.error(f"Failed to load YOLO model {model_path}: {e}")
            raise
    
    def predict(self, frame: np.ndarray, conf_threshold: float = 0.7, model_path: str = "yolov8m.pt") -> List[Dict[str, Any]]:
        """Run inference on frame and return detections using the specified model"""
        if not self.initialized or model_path not in self.models:
            raise RuntimeError(f"YOLO model {model_path} not initialized")
        
        with self.model_lock:
            try:
                results = self.models[model_path](frame, conf=conf_threshold)
                return self._parse_results(results, model_path)
            except Exception as e:
                logger.error(f"YOLO inference failed for model {model_path}: {e}")
                return []
    
    def _parse_results(self, results, model_path: str) -> List[Dict[str, Any]]:
        """Parse YOLO results into standardized format"""
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    class_name = self.models[model_path].names[class_id]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name,
                        'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
                    }
                    detections.append(detection)
        
        return detections