### File: app/main.py
from fastapi import FastAPI
from app.api.v1.endpoints import api_router
from app.core.config import settings
from app.core.yolo_manager import YOLOManager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

@app.on_event("startup")
async def startup_event():
    """Initialize YOLO model on startup"""
    try:
        yolo_manager = YOLOManager()
        await yolo_manager.initialize_model()
        logger.info("YOLO model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {e}")
        raise

app.include_router(api_router, prefix="/api/v1")

### File: app/core/config.py
from pydantic import BaseSettings
from typing import Dict, List, Any

class Settings(BaseSettings):
    PROJECT_NAME: str = "Video Analytics System"
    VERSION: str = "1.0.0"
    
    # MongoDB Configuration
    MONGO_CREDENTIALS: Dict[str, str] = {
        "connection_string": "mongodb+srv://Tarun7053:<db_password>@cluster0.krmwp.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        "password": "Tarun@1030",
        "db_name": "test"
    }
    
    # Rule Configuration
    RULE_DATA: List[Dict[str, Any]] = [
        {"id": 1, "rule": "Intrusion"},
        {"id": 2, "rule": "Camera Tempering"},
        {"id": 3, "rule": "Entry Exit"}
    ]
    
    # YOLO Configuration
    YOLO_MODEL_PATH: str = "yolov8m.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.7
    FRAME_INTERVAL: int = 1
    
    # Processing Configuration
    OUTPUT_VIDEO_DIR: str = "output_videos"
    
    class Config:
        env_file = ".env"

settings = Settings()

### File: app/core/yolo_manager.py
import asyncio
import threading
from typing import Optional, List, Dict, Any
from ultralytics import YOLO
import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class YOLOManager:
    """Singleton YOLO model manager for shared inference across all cameras"""
    
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
            self.model: Optional[YOLO] = None
            self.model_lock = threading.Lock()
            self.initialized = False
    
    async def initialize_model(self, model_path: str = "yolov8m.pt"):
        """Initialize YOLO model"""
        try:
            with self.model_lock:
                if not self.initialized:
                    self.model = YOLO(model_path)
                    self.initialized = True
                    logger.info(f"YOLO model loaded: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def predict(self, frame: np.ndarray, conf_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Run inference on frame and return detections"""
        if not self.initialized or self.model is None:
            raise RuntimeError("YOLO model not initialized")
        
        with self.model_lock:
            try:
                results = self.model(frame, conf=conf_threshold)
                return self._parse_results(results)
            except Exception as e:
                logger.error(f"YOLO inference failed: {e}")
                return []
    
    def _parse_results(self, results) -> List[Dict[str, Any]]:
        """Parse YOLO results into standardized format"""
        detections = []
        
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf)
                    class_id = int(box.cls)
                    class_name = self.model.names[class_id]
                    
                    detection = {
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'class_id': class_id,
                        'class_name': class_name,
                        'centroid': [(x1 + x2) // 2, (y1 + y2) // 2]
                    }
                    detections.append(detection)
        
        return detections

### File: app/core/spatial_intelligence.py
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple

class SpatialIntelligence:
    """Spatial analysis utilities for video analytics"""
    
    @staticmethod
    def calculate_centroid(bbox: List[int]) -> Tuple[int, int]:
        """Calculate centroid from bounding box"""
        x1, y1, x2, y2 = bbox
        return (x1 + x2) // 2, (y1 + y2) // 2
    
    @staticmethod
    def is_inside_roi(point: Tuple[int, int], roi: np.ndarray) -> bool:
        """Check if point is inside ROI polygon"""
        return cv2.pointPolygonTest(roi, point, False) >= 0
    
    @staticmethod
    def bbox_roi_intersection(bbox: List[int], roi: np.ndarray) -> bool:
        """Check if bounding box intersects with ROI"""
        x1, y1, x2, y2 = bbox
        
        # Check corners and bottom line
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        bottom_line = [(x1, y2), (x2, y2)]
        
        # Check if any corner or bottom line point is inside ROI
        for point in corners + bottom_line:
            if SpatialIntelligence.is_inside_roi(point, roi):
                return True
        
        return False
    
    @staticmethod
    def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate Intersection over Union between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union area
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    @staticmethod
    def analyze_detections(detections: List[Dict[str, Any]], roi_configs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze detections and extract spatial intelligence"""
        spatial_data = {
            'detections': detections,
            'person_detections': [],
            'centroids': [],
            'roi_intersections': {},
            'detection_count': len(detections)
        }
        
        # Filter person detections
        for detection in detections:
            if detection['class_name'] == 'person':
                spatial_data['person_detections'].append(detection)
                spatial_data['centroids'].append(detection['centroid'])
        
        # ROI analysis if ROI configs provided
        if roi_configs:
            for roi_name, roi_points in roi_configs.items():
                roi_array = np.array(roi_points, dtype=np.int32)
                spatial_data['roi_intersections'][roi_name] = []
                
                for detection in spatial_data['person_detections']:
                    centroid = detection['centroid']
                    bbox = detection['bbox']
                    
                    # Check both centroid and bbox intersection
                    centroid_inside = SpatialIntelligence.is_inside_roi(centroid, roi_array)
                    bbox_intersects = SpatialIntelligence.bbox_roi_intersection(bbox, roi_array)
                    
                    if centroid_inside or bbox_intersects:
                        spatial_data['roi_intersections'][roi_name].append({
                            'detection': detection,
                            'centroid_inside': centroid_inside,
                            'bbox_intersects': bbox_intersects
                        })
        
        return spatial_data

### File: app/core/camera_processor.py
import cv2
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from app.core.yolo_manager import YOLOManager
from app.core.spatial_intelligence import SpatialIntelligence
from app.core.config import settings

logger = logging.getLogger(__name__)

class CameraProcessor:
    """Main camera processing engine that handles single camera with multiple rules"""
    
    def __init__(self, camera_name: str, stream_url: str, rules: List[str], 
                 session_id: str, mongo_credentials: Dict[str, str], 
                 yolo_manager: Optional[YOLOManager] = None):
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.rules = rules
        self.session_id = session_id
        self.mongo_credentials = mongo_credentials
        
        # Use provided YOLO manager or create new one
        self.yolo_manager = yolo_manager if yolo_manager else YOLOManager()
        self.spatial_intelligence = SpatialIntelligence()
        
        # Rule function mapping
        self.rule_functions = {
            'Intrusion': self._process_intrusion_rule,
            'Camera Tempering': self._process_camera_tempering_rule,
            'Entry Exit': self._process_entry_exit_rule
        }
        
        # Rule-specific state
        self.rule_states = {
            'Intrusion': {'prev_intrusion_detected': False},
            'Camera Tempering': {'prev_tampered': False, 'prev_reason': '', 'movement_history': []},
            'Entry Exit': {'tracker': None, 'people_entering': {}, 'entering': set(), 
                         'people_exiting': {}, 'exiting': set(), 'last_logged_enter': 0, 'last_logged_exit': 0}
        }
        
        # Camera-specific ROI configurations
        self.roi_configs = {
            'intrusion_roi': [[400, 300], [750, 300], [750, 800], [400, 800]]
        }
        
        self.frame_count = 0
        self.is_processing = False
        self._loop = asyncio.get_event_loop()
        self.video_writers = {}  # Dictionary to store video writers for each rule
        self.output_dir = settings.OUTPUT_VIDEO_DIR
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def read_frame_async(self, cap):
        """Read frame asynchronously using ThreadPoolExecutor"""
        with ThreadPoolExecutor() as executor:
            ret, frame = await self._loop.run_in_executor(executor, cap.read)
        return ret, frame
    
    async def initialize_video_writer(self, rule_name: str, cap):
        """Initialize video writer for a specific rule"""
        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0  # Default to 30 if FPS not available
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Generate unique output filename for the rule
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                self.output_dir,
                f"{self.camera_name}_{rule_name.replace(' ', '_')}_{self.session_id}_{timestamp}.mp4"
            )
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
            video_writer = cv2.VideoWriter(
                output_filename, fourcc, fps, (width, height)
            )
            
            if not video_writer.isOpened():
                logger.error(f"Failed to initialize video writer for {output_filename}")
                return None
                
            logger.info(f"Video writer initialized for rule {rule_name}: {output_filename}")
            return video_writer
            
        except Exception as e:
            logger.error(f"Error initializing video writer for rule {rule_name}: {e}")
            return None
    
    async def write_frame_async(self, rule_name: str, frame: np.ndarray):
        """Write frame to video file for a specific rule asynchronously"""
        if rule_name in self.video_writers and self.video_writers[rule_name]:
            try:
                with ThreadPoolExecutor() as executor:
                    await self._loop.run_in_executor(executor, self.video_writers[rule_name].write, frame)
            except Exception as e:
                logger.error(f"Error writing frame for rule {rule_name}: {e}")
    
    async def start_processing(self):
        """Start camera processing with all assigned rules"""
        self.is_processing = True
        logger.info(f"Starting processing for camera {self.camera_name} with rules: {self.rules}")
        
        try:
            await self._process_camera_stream()
        except asyncio.CancelledError:
            logger.info(f"Processing cancelled for camera {self.camera_name}")
            raise
        except Exception as e:
            logger.error(f"Error processing camera {self.camera_name}: {e}")
        finally:
            self._cleanup()
    
    async def _process_camera_stream(self):
        """Main processing loop for camera stream"""
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.stream_url}")
        
        logger.info(f"Camera {self.camera_name} stream opened successfully")
        
        try:
            # Initialize video writers for each rule
            for rule_name in self.rules:
                if rule_name in self.rule_functions:
                    self.video_writers[rule_name] = await self.initialize_video_writer(rule_name, cap)
            
            while self.is_processing:
                ret, frame = await self.read_frame_async(cap)
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_name}")
                    break
                
                self.frame_count += 1
                
                # Process frame at specified interval
                if self.frame_count % settings.FRAME_INTERVAL == 0:
                    await self._process_frame(frame)
                    
                # Small sleep to prevent CPU overload
                await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error in camera processing loop: {e}")
        finally:
            cap.release()
            self._cleanup()
            logger.info(f"Camera {self.camera_name} stream closed")
    
    def _cleanup(self):
        """Clean up resources"""
        self.is_processing = False
        for rule_name, video_writer in self.video_writers.items():
            if video_writer:
                try:
                    video_writer.release()
                    logger.info(f"Video writer released for camera {self.camera_name}, rule {rule_name}")
                except Exception as e:
                    logger.error(f"Error releasing video writer for rule {rule_name}: {e}")
        self.video_writers.clear()
    
    async def _process_frame(self, frame: np.ndarray):
        """Process single frame with YOLO inference and rule execution"""
        try:
            # Run YOLO inference once
            detections = self.yolo_manager.predict(frame, settings.YOLO_CONFIDENCE_THRESHOLD)
            
            # Extract spatial intelligence
            spatial_data = self.spatial_intelligence.analyze_detections(
                detections, self.roi_configs
            )
            
            # Execute each rule and save to its own video
            for rule_name in self.rules:
                if rule_name in self.rule_functions:
                    # Create a copy of the frame for this rule
                    frame_copy = frame.copy()
                    await self.rule_functions[rule_name](frame_copy, spatial_data)
                    # Write the processed frame to the rule-specific video
                    await self.write_frame_async(rule_name, frame_copy)
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera_name}: {e}")
    
    async def _process_intrusion_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        """Process intrusion detection rule"""
        from app.rules.intrusion import process_intrusion_rule
        
        await process_intrusion_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Intrusion']
        )
    
    async def _process_camera_tempering_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        """Process camera tempering rule"""
        from app.rules.camera_tempering import process_camera_tempering_rule
        
        await process_camera_tempering_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Camera Tempering']
        )
    
    async def _process_entry_exit_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        """Process entry/exit counting rule"""
        from app.rules.entry_exit import process_entry_exit_rule
        
        await process_entry_exit_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Entry Exit']
        )
    
    def stop_processing(self):
        """Stop camera processing"""
        self.is_processing = False
        logger.info(f"Stopping processing for camera {self.camera_name}")


### File: app/db/database.py
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from urllib.parse import quote_plus
from typing import Tuple, Optional

# Global MongoDB client and collections
mongo_client = None
db = None
_collection_cam_details = None
mongo_credentials = {
    "connection_string": None,
    "password": None,
    "db_name": None
}

def set_mongo_credentials(connection_string: str, password: str, db_name: str):
    """Store MongoDB credentials globally."""
    mongo_credentials["connection_string"] = connection_string
    mongo_credentials["password"] = password
    mongo_credentials["db_name"] = db_name

def get_mongo_client(connection_string: str, password: str, db_name: str) -> Tuple[MongoClient, object]:
    """Initialize and return a MongoDB client connection dynamically."""
    global mongo_client, db, _collection_cam_details

    try:
        # Ensure password is URL-encoded
        encoded_password = quote_plus(password)
        
        # Replace placeholder with the actual password
        mongo_uri = connection_string.replace("<db_password>", encoded_password)
        
        # Create a new client
        new_client = MongoClient(mongo_uri)
        
        # Verify authentication
        try:
            new_client.admin.command("ping")
        except OperationFailure as auth_error:
            raise Exception("Authentication failed! Check your username and password.") from auth_error
        
        # Assign new connection if successful
        mongo_client = new_client
        db = mongo_client[db_name]
        
        # Create collections if they do not exist
        _collection_cam_details = db["cam_details"]
        
        return mongo_client, db
        
    except ConnectionFailure:
        raise Exception("Failed to connect to MongoDB. Check your network or MongoDB URI.")
    except Exception as e:
        raise Exception(f"Error: {e}")

def set_mongo_client(client: MongoClient, database):
    """Explicitly set the global MongoDB client and database after reconnection or disconnection."""
    global mongo_client, db, _collection_cam_details
    
    mongo_client = client
    db = database
    
    if db is not None:
        _collection_cam_details = db["cam_details"]
    else:
        _collection_cam_details = None

def get_collection_cam_details():
    """Return the 'cam_details' collection if initialized."""
    if _collection_cam_details is None:
        raise Exception("Database not initialized. Connect first.")
    return _collection_cam_details

def save_cam(camera_name: str, module_name: list, location: str, stream_url: str):
    """Save camera details to the 'cam_details' collection."""
    try:
        collection = get_collection_cam_details()
        camera_data = {
            "camera_name": camera_name,
            "module_names": module_name,
            "location": location,
            "stream_url": stream_url
        }
        result = collection.insert_one(camera_data)
        return result.inserted_id
    except OperationFailure as e:
        raise Exception(f"Failed to save camera details: {str(e)}")
    except Exception as e:
        raise Exception(f"Error saving camera details: {str(e)}")

### File: app/rules/intrusion.py
# app/rules/intrusion.py
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
import logging

logger = logging.getLogger(__name__)

async def process_intrusion_rule(frame: np.ndarray, spatial_data: Dict[str, Any], 
                                camera_name: str, session_id: str, 
                                mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process intrusion detection rule with shared detection data"""
    
    try:
        # Get collection directly (database is already initialized)
        collection = get_collection("intrusion")
        
        # Get ROI data
        roi_data = [[400, 300], [750, 300], [750, 800], [400, 800]]
        roi_coordinates_int = np.array(roi_data, dtype=np.int32)
        
        # Draw ROI on frame
        cv2.polylines(frame, [roi_coordinates_int], isClosed=True, color=(0, 255, 255), thickness=2)
        
        # Check for intrusion using spatial data
        intrusion_detected = False
        
        # Check if any person detection intersects with ROI
        if 'roi_intersections' in spatial_data and 'intrusion_roi' in spatial_data['roi_intersections']:
            roi_intersections = spatial_data['roi_intersections']['intrusion_roi']
            if roi_intersections:
                intrusion_detected = True
        
        # Alternative: Check person detections manually
        if not intrusion_detected:
            for detection in spatial_data.get('person_detections', []):
                if detection['confidence'] > 0.7:
                    bbox = detection['bbox']
                    centroid = detection['centroid']
                    
                    # Check if centroid or bottom line is inside ROI
                    x1, y1, x2, y2 = bbox
                    bottom_line_points = [(x1, y2), (x2, y2)]
                    
                    centroid_inside = cv2.pointPolygonTest(roi_coordinates_int, centroid, False) >= 0
                    bottom_line_inside = any(
                        cv2.pointPolygonTest(roi_coordinates_int, point, False) >= 0 
                        for point in bottom_line_points
                    )
                    
                    if centroid_inside or bottom_line_inside:
                        intrusion_detected = True
                        
                        # Draw detection
                        color = (0, 0, 255)  # Red for intrusion
                        label = "Intrusion!"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                        break
        
        # Save to MongoDB if intrusion detected and state changed
        prev_intrusion_detected = rule_state.get('prev_intrusion_detected', False)
        
        if intrusion_detected and not prev_intrusion_detected:
            data = {
                "timestamp": datetime.now(),
                "intrusion": intrusion_detected,
                "session_id": session_id,
                "camera_name": camera_name
            }
            try:
                collection.insert_one(data)
                logger.info(f"Intrusion detected and saved for camera {camera_name}")
            except Exception as e:
                logger.error(f"Error inserting intrusion data to DB: {e}")
        
        # Update rule state
        rule_state['prev_intrusion_detected'] = intrusion_detected
        
    except Exception as e:
        logger.error(f"Error in intrusion rule processing: {e}")
        # Don't re-raise to avoid breaking the processing loop

### File: app/rules/camera_tempering.py
# app/rules/camera_tempering.py
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any
from collections import deque
from app.db.database import get_collection
import logging

logger = logging.getLogger(__name__)

def detect_occlusion(frame: np.ndarray, brightness_threshold: int = 40) -> bool:
    """Detect camera occlusion based on brightness"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < brightness_threshold

def detect_movement(prev_frame: np.ndarray, curr_frame: np.ndarray, diff_threshold: int = 10000) -> bool:
    """Detect camera movement based on frame difference"""
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = np.sum(gray_diff) // 10000
    return score > diff_threshold

async def process_camera_tempering_rule(frame: np.ndarray, spatial_data: Dict[str, Any],
                                       camera_name: str, session_id: str,
                                       mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process camera tempering detection rule"""
    
    try:
        # Get collection directly (database is already initialized)
        collection = get_collection("camera_tempering")
        
        # Initialize state variables
        if 'movement_history' not in rule_state:
            rule_state['movement_history'] = deque(maxlen=5)
        if 'first_movement_timestamp' not in rule_state:
            rule_state['first_movement_timestamp'] = None
        if 'last_inserted_reason' not in rule_state:
            rule_state['last_inserted_reason'] = None
        if 'prev_frame' not in rule_state:
            rule_state['prev_frame'] = frame.copy()
            return
        
        # Check for tampering
        current_tampered = False
        current_reason = ""
        
        # Check for occlusion
        if detect_occlusion(frame):
            current_tampered = True
            current_reason = "occlusion"
            logger.info(f"Camera occlusion detected for {camera_name}")
        else:
            # Check for movement
            movement_detected = detect_movement(rule_state['prev_frame'], frame)
            rule_state['movement_history'].append(movement_detected)
            
            # Track first timestamp for movement detection
            if movement_detected and len(rule_state['movement_history']) >= 2:
                if not rule_state['movement_history'][-2]:
                    rule_state['first_movement_timestamp'] = datetime.now()
            
            # Check if movement detected in at least 2 of last 5 frames
            if len(rule_state['movement_history']) >= 5 and sum(rule_state['movement_history']) >= 2:
                current_tampered = True
                current_reason = "camera movement"
                logger.info(f"Camera movement detected for {camera_name}")
        
        # Save to DB if tampering state changed
        prev_tampered = rule_state.get('prev_tampered', False)
        prev_reason = rule_state.get('prev_reason', "")
        
        if (not prev_tampered and current_tampered) or (current_tampered and current_reason != prev_reason):
            # Add visual indicator
            cv2.putText(frame, f"TAMPERING DETECTED: {current_reason.upper()}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Use first_movement_timestamp for movement, otherwise current timestamp
            timestamp = rule_state['first_movement_timestamp'] if current_reason == "camera movement" else datetime.now()
            
            # Only insert if reason is different from last inserted
            if current_reason != rule_state['last_inserted_reason']:
                data = {
                    "timestamp": timestamp,
                    "camera_name": camera_name,
                    "session_id": session_id,
                    "cam_temp": True,
                    "reason": current_reason
                }
                try:
                    collection.insert_one(data)
                    rule_state['last_inserted_reason'] = current_reason
                    logger.info(f"Camera tempering saved for {camera_name}: {current_reason}")
                except Exception as e:
                    logger.error(f"Error inserting camera tempering data to DB: {e}")
        
        # Update rule state
        rule_state['prev_tampered'] = current_tampered
        rule_state['prev_reason'] = current_reason
        rule_state['prev_frame'] = frame.copy()
        
    except Exception as e:
        logger.error(f"Error in camera tempering rule processing: {e}")
        # Don't re-raise to avoid breaking the processing loop

### File: app/rules/entry_exit.py
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, Any, Set
from app.db.database import get_mongo_client
from app.rules.utils.tracker import Tracker
import logging

logger = logging.getLogger(__name__)

# Entry/Exit areas
AREA1 = [(312, 388), (289, 390), (474, 469), (497, 462)]  # Entry area
AREA2 = [(279, 392), (250, 397), (423, 477), (454, 469)]  # Exit area

async def process_entry_exit_rule(frame: np.ndarray, spatial_data: Dict[str, Any],
                                 camera_name: str, session_id: str,
                                 mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process entry/exit counting rule"""
    
    try:
        # Initialize tracker if not exists
        if 'tracker' not in rule_state or rule_state['tracker'] is None:
            rule_state['tracker'] = Tracker()
        
        # Initialize other state variables
        if 'people_entering' not in rule_state:
            rule_state['people_entering'] = {}
        if 'entering' not in rule_state:
            rule_state['entering'] = set()
        if 'people_exiting' not in rule_state:
            rule_state['people_exiting'] = {}
        if 'exiting' not in rule_state:
            rule_state['exiting'] = set()
        if 'last_logged_enter' not in rule_state:
            rule_state['last_logged_enter'] = 0
        if 'last_logged_exit' not in rule_state:
            rule_state['last_logged_exit'] = 0
        
        # Get MongoDB connection
        client, database = get_mongo_client(
            mongo_credentials["connection_string"],
            mongo_credentials["password"],
            mongo_credentials["db_name"]
        )
        collection = database["entry_exit"]
        
        # Prepare bounding boxes for tracker
        bbox_list = []
        for detection in spatial_data.get('person_detections', []):
            bbox = detection['bbox']
            bbox_list.append(bbox)
        
        # Update tracker
        bbox_id = rule_state['tracker'].update(bbox_list)
        
        # Draw regions
        cv2.polylines(frame, [np.array(AREA1, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, "Entry", (504, 471), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        
        cv2.polylines(frame, [np.array(AREA2, np.int32)], True, (255, 0, 0), 2)
        cv2.putText(frame, "Exit", (466, 485), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
        
        for bbox in bbox_id:
            x1, y1, x2, y2, track_id = bbox
            
            # Process entering (exit area to entry area)
            _process_person_entering(frame, x1, y1, x2, y2, track_id, rule_state)
            
            # Process exiting (entry area to exit area)
            _process_person_exiting(frame, x1, y1, x2, y2, track_id, rule_state)
        
        # Insert data to DB if counts changed
        _insert_data_to_db(collection, camera_name, session_id, rule_state)
        
        # Display counts on frame
        cv2.putText(frame, f"Entering: {len(rule_state['entering'])}", (20, 44),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Exiting: {len(rule_state['exiting'])}", (20, 82),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        
    except Exception as e:
        logger.error(f"Error in entry/exit rule processing: {e}")

def _process_person_entering(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                           track_id: int, rule_state: Dict[str, Any]):
    """Process person entering (going from exit_area to entry_area)"""
    # Check if person is in exit area (area2)
    results = cv2.pointPolygonTest(np.array(AREA2, np.int32), (x2, y2), False)
    if results >= 0:
        rule_state['people_entering'][track_id] = (x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Check if person moved to entry area (area1)
    if track_id in rule_state['people_entering']:
        results1 = cv2.pointPolygonTest(np.array(AREA1, np.int32), (x2, y2), False)
        if results1 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (x2, y2), 4, (255, 0, 255), -1)
            cv2.putText(frame, "enter", (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"ID: {track_id}", (x1+65, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            rule_state['entering'].add(track_id)

def _process_person_exiting(frame: np.ndarray, x1: int, y1: int, x2: int, y2: int, 
                          track_id: int, rule_state: Dict[str, Any]):
    """Process person exiting (going from entry_area to exit_area)"""
    # Check if person is in entry area (area1)
    results2 = cv2.pointPolygonTest(np.array(AREA1, np.int32), (x2, y2), False)
    if results2 >= 0:
        rule_state['people_exiting'][track_id] = (x2, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Check if person moved to exit area (area2)
    if track_id in rule_state['people_exiting']:
        results3 = cv2.pointPolygonTest(np.array(AREA2, np.int32), (x2, y2), False)
        if results3 >= 0:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.circle(frame, (x2, y2), 4, (255, 0, 255), -1)
            cv2.putText(frame, "exit", (x1, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"ID: {track_id}", (x1+55, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
            rule_state['exiting'].add(track_id)

def _insert_data_to_db(collection, camera_name: str, session_id: str, rule_state: Dict[str, Any]):
    """Insert data to MongoDB if counts change"""
    entry_count = len(rule_state['entering'])
    exit_count = len(rule_state['exiting'])
    
    if entry_count != rule_state['last_logged_enter'] or exit_count != rule_state['last_logged_exit']:
        current_entry = entry_count - rule_state['last_logged_enter']
        current_exit = exit_count - rule_state['last_logged_exit']
        
        try:
            collection.insert_one({
                "camera_name": camera_name,
                "session_id": session_id,
                "timestamp": datetime.now(),
                "entry_count": current_entry,
                "exit_count": current_exit,
                "total_entries": entry_count,
                "total_exits": exit_count
            })
            logger.info(f"Entry/Exit data saved for camera {camera_name}: +{current_entry} entries, +{current_exit} exits")
        except Exception as e:
            logger.error(f"Error inserting entry/exit data to DB: {e}")
        
        rule_state['last_logged_enter'] = entry_count
        rule_state['last_logged_exit'] = exit_count

### File: app/rules/tracker.py
import math
from typing import List, Tuple, Dict, Any

class Tracker:
    """Object tracker for maintaining consistent IDs across frames"""
    
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
    
    def update(self, objects_rect: List[List[int]]) -> List[List[int]]:
        """Update tracker with new detections and return tracked objects with IDs"""
        objects_bbs_ids = []
        
        # Get center point of new objects
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + w) // 2
            cy = (y + h) // 2
            
            # Find if object already exists
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                
                if dist < 35:  # Threshold for same object
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            
            # New object is detected
            if not same_object_detected:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
        
        # Clean up center points dict by removing IDs not found in current frame
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        
        # Update dictionary with IDs not found in current frame
        self.center_points = new_center_points.copy()
        
        return objects_bbs_ids

### File: app/api/v1/endpoints.py
from fastapi import APIRouter
from app.api.v1.routes import camera, rules, processing

api_router = APIRouter()

api_router.include_router(camera.router, prefix="/cameras", tags=["Cameras"])
api_router.include_router(rules.router, prefix="/rules", tags=["Rules"])
api_router.include_router(processing.router, prefix="/processing", tags=["Processing"])

### File: app/api/v1/routes/camera.py
from fastapi import APIRouter, HTTPException
from typing import List
from app.schemas.camera import CameraCreate, CameraOut
from app.db.database import save_cam, get_collection_cam_details
from app.core.config import settings

router = APIRouter()

@router.post("/", response_model=dict)
async def create_camera(camera: CameraCreate):
    """Create a new camera configuration"""
    try:
        camera_id = save_cam(
            camera_name=camera.camera_name,
            module_name=camera.module_names,
            location=camera.location,
            stream_url=camera.stream_url
        )
        return {
            "message": "Camera created successfully",
            "camera_id": str(camera_id),
            "camera_name": camera.camera_name
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/", response_model=List[CameraOut])
async def get_cameras():
    """Get all cameras"""
    try:
        collection = get_collection_cam_details()
        cameras = list(collection.find({}))
        
        # Convert MongoDB ObjectId to string
        for camera in cameras:
            camera['_id'] = str(camera['_id'])
        
        return cameras
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{camera_name}", response_model=CameraOut)
async def get_camera(camera_name: str):
    """Get specific camera by name"""
    try:
        collection = get_collection_cam_details()
        camera = collection.find_one({"camera_name": camera_name})
        
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        camera['_id'] = str(camera['_id'])
        return camera
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

### File: app/api/v1/routes/rules.py
from fastapi import APIRouter
from typing import List
from app.schemas.rule import RuleOut
from app.core.config import settings

router = APIRouter()

@router.get("/", response_model=List[RuleOut])
async def get_rules():
    """Get all available rules"""
    return settings.RULE_DATA

@router.get("/{rule_id}", response_model=RuleOut)
async def get_rule(rule_id: int):
    """Get specific rule by ID"""
    rule = next((rule for rule in settings.RULE_DATA if rule["id"] == rule_id), None)
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return rule

### File: app/api/v1/routes/processing.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import logging
import uuid
from app.schemas.processing import ProcessingRequest, ProcessingResponse
from app.core.config import settings
from app.db.database import initialize_database, get_collection_cam_details
from app.core.camera_processor import CameraProcessor
from app.core.yolo_manager import YOLOManager

router = APIRouter()
logger = logging.getLogger(__name__)

# Global state for async tasks
active_tasks = {}
task_status = {}
shared_yolo_manager = None

async def init_mongo_and_yolo():
    """Initialize MongoDB and shared YOLO manager"""
    global shared_yolo_manager
    try:
        # Initialize database connection
        await initialize_database(
            settings.MONGO_CREDENTIALS["connection_string"],
            settings.MONGO_CREDENTIALS["password"],
            settings.MONGO_CREDENTIALS["db_name"]
        )
        logger.info("MongoDB connection established successfully!")

        # Initialize shared YOLO manager
        shared_yolo_manager = YOLOManager()
        await shared_yolo_manager.initialize_model(settings.YOLO_MODEL_PATH)
        logger.info("Shared YOLO manager initialized successfully")
        
        # Fetch camera details
        collection = get_collection_cam_details()
        documents = collection.find()
        return [{**doc, "_id": str(doc["_id"])} for doc in documents]
        
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")

async def validate_camera_stream(stream_url, camera_name):
    """Validate if camera stream is accessible asynchronously"""
    try:
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            cap = await loop.run_in_executor(executor, cv2.VideoCapture, stream_url)
            if not cap.isOpened():
                logger.error(f"Cannot open stream for camera {camera_name}: {stream_url}")
                return False
            
            ret, _ = await loop.run_in_executor(executor, cap.read)
            await loop.run_in_executor(executor, cap.release)
            
            if not ret:
                logger.error(f"Cannot read frame from camera {camera_name}: {stream_url}")
                return False
                
            return True
    except Exception as e:
        logger.error(f"Error validating stream for camera {camera_name}: {e}")
        return False

async def process_camera_task(session_id: str, camera_name: str, stream_url: str, 
                            rules: List[str], mongo_credentials: Dict[str, str]):
    """Process a single camera asynchronously"""
    try:
        logger.info(f"Processing started for camera {camera_name} with session {session_id}")
        
        processor = CameraProcessor(
            camera_name=camera_name,
            stream_url=stream_url,
            rules=rules,
            session_id=session_id,
            mongo_credentials=mongo_credentials,
            yolo_manager=shared_yolo_manager
        )
        
        task_status[session_id] = {
            "camera_name": camera_name,
            "status": "running",
            "message": f"Processing started for {camera_name}"
        }
        
        await processor.start_processing()
        
        task_status[session_id] = {
            "camera_name": camera_name,
            "status": "completed",
            "message": f"Processing finished successfully for {camera_name}"
        }
        
    except Exception as e:
        error_msg = f"Error processing camera {camera_name}: {str(e)}"
        logger.error(error_msg)
        task_status[session_id] = {
            "camera_name": camera_name,
            "status": "error",
            "message": error_msg
        }
        raise

@router.post("/start", response_model=List[ProcessingResponse])
async def start_processing(_: None = Depends(init_mongo_and_yolo)):
    """Start video processing for all cameras using asyncio tasks"""
    global active_tasks, task_status, shared_yolo_manager
    
    logger.info("Starting async processing for all cameras from database")
    
    cam_info = await init_mongo_and_yolo()
    if not cam_info:
        raise HTTPException(status_code=400, detail="No camera details found. Please add cameras first.")
    
    responses = []
    valid_cameras = []
    
    # Validate all camera streams
    for cam in cam_info:
        camera_name = cam.get("camera_name")
        stream_url = cam.get("stream_url")
        module_names = cam.get("module_names", [])
        
        if not camera_name or not stream_url:
            logger.warning(f"Skipping camera with missing data: {cam}")
            continue
            
        if not module_names:
            logger.warning(f"Skipping camera {camera_name} - no rules defined")
            continue
        
        if not await validate_camera_stream(stream_url, camera_name):
            logger.error(f"Skipping camera {camera_name} - stream not accessible")
            continue
            
        valid_cameras.append(cam)
        logger.info(f"Validated camera: {camera_name} with rules: {module_names}")
    
    if not valid_cameras:
        raise HTTPException(status_code=400, detail="No valid camera streams found")
    
    # Start async tasks for each camera
    tasks = []
    for cam in valid_cameras:
        camera_name = cam["camera_name"]
        stream_url = cam["stream_url"]
        module_names = cam["module_names"]
        
        session_id = str(uuid.uuid4())
        
        task = asyncio.create_task(
            process_camera_task(
                session_id,
                camera_name,
                stream_url,
                module_names,
                settings.MONGO_CREDENTIALS
            )
        )
        
        active_tasks[session_id] = task
        task_status[session_id] = {
            "camera_name": camera_name,
            "status": "started",
            "message": f"Processing started for {camera_name}"
        }
        
        responses.append(ProcessingResponse(
            session_id=session_id,
            message=f"Processing started successfully for {camera_name}",
            cameras=[camera_name]
        ))
    
    # Run all tasks concurrently
    await asyncio.gather(*active_tasks.values(), return_exceptions=True)
    
    logger.info(f"Started {len(responses)} camera async tasks")
    return responses

@router.post("/stop/{session_id}")
async def stop_processing(session_id: str):
    """Stop video processing session for a specific camera"""
    global active_tasks, task_status
    
    if session_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Session not found")
    
    task = active_tasks[session_id]
    camera_name = task_status.get(session_id, {}).get("camera_name", "unknown")
    
    # Cancel the task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Clean up
    if session_id in active_tasks:
        del active_tasks[session_id]
    if session_id in task_status:
        del task_status[session_id]
    
    return {"message": f"Processing session {session_id} for camera {camera_name} stopped successfully"}

@router.post("/stop-all")
async def stop_all_processing():
    """Stop all active processing sessions"""
    global active_tasks, task_status
    
    stopped_sessions = []
    
    for session_id, task in list(active_tasks.items()):
        try:
            camera_name = task_status.get(session_id, {}).get("camera_name", "unknown")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
                
            stopped_sessions.append({
                "session_id": session_id,
                "camera": camera_name
            })
        except Exception as e:
            logger.error(f"Error stopping session {session_id}: {e}")
    
    # Clear all tasks
    active_tasks.clear()
    task_status.clear()
    
    return {
        "message": f"Stopped {len(stopped_sessions)} processing sessions",
        "stopped_sessions": stopped_sessions
    }

@router.get("/sessions")
async def get_active_sessions():
    """Get all active processing sessions with their status"""
    global active_tasks, task_status
    
    sessions = {}
    for session_id, task in active_tasks.items():
        status_info = task_status.get(session_id, {
            "camera_name": "unknown",
            "status": "running",
            "message": "Processing running"
        })
        
        sessions[session_id] = {
            "camera": status_info["camera_name"],
            "status": "running" if not task.done() else "completed",
            "message": status_info["message"]
        }
    
    return sessions

@router.get("/session-results/{session_id}")
async def get_session_results(session_id: str):
    """Get results for a specific session"""
    global task_status
    
    if session_id not in task_status:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": session_id,
        "results": [task_status[session_id]]
    }

async def cleanup_async_tasks():
    """Cleanup all async tasks on application shutdown"""
    logger.info("Cleaning up all async tasks")
    await stop_all_processing()

    
### File: app/schemas/camera.py
from pydantic import BaseModel
from typing import List, Optional

class CameraCreate(BaseModel):
    camera_name: str
    module_names: List[str]
    location: str
    stream_url: str

class CameraOut(BaseModel):
    _id: Optional[str] = None
    camera_name: str
    module_names: List[str]
    location: str
    stream_url: str

### File: app/schemas/rule.py
from pydantic import BaseModel

class RuleOut(BaseModel):
    id: int
    rule: str

### File: app/schemas/processing.py
from pydantic import BaseModel
from typing import Dict, List

class ProcessingRequest(BaseModel):
    camera_names: List[str]
    camera_rules: Dict[str, List[str]]  # camera_name -> list of rules

class ProcessingResponse(BaseModel):
    session_id: str
    message: str
    cameras: List[str]

import cv2
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from collections import deque
from app.core.yolo_manager import YOLOManager
from app.core.spatial_intelligence import SpatialIntelligence
from app.core.config import settings
from app.rules.fall import CentroidTracker

logger = logging.getLogger(__name__)

class CameraProcessor:
    """Main camera processing engine that handles single camera with multiple rules"""
    
    def __init__(self, camera_name: str, stream_url: str, rules: List[str], 
                 session_id: str, mongo_credentials: Dict[str, str], 
                 yolo_manager: Optional[YOLOManager] = None):
        self.camera_name = camera_name
        self.stream_url = stream_url
        self.rules = rules
        self.session_id = session_id
        self.mongo_credentials = mongo_credentials
        
        # Use provided YOLO manager or create new one
        self.yolo_manager = yolo_manager if yolo_manager else YOLOManager()
        self.spatial_intelligence = SpatialIntelligence()
        
        # Rule functions
        self.rule_functions = {
            'Intrusion': self._process_intrusion_rule,
            'Camera Tempering': self._process_camera_tempering_rule,
            'Entry Exit': self._process_entry_exit_rule,
            'Occupancy Monitoring': self._process_occupancy_monitoring_rule,
            'Employee Unavailability': self._process_employee_unavailability_rule,
            'Mobile Usage': self._process_mobile_usage_rule,
            'Attended/Unattended Time': self._process_attended_unattended_time_rule,
            'Customer Staff Ratio': self._process_customer_staff_ratio_rule,
            'Dwell Time': self._process_dwell_time_rule,
            'Fall/Slip': self._process_fall_rule,
            'Billing Alerts': self._process_billing_alerts_rule,
            'Billing Counter': self._process_billing_counter_rule
        }

        # Rule states
        self.rule_states = {
            'Intrusion': {'prev_intrusion_detected': False},
            'Camera Tempering': {
                'prev_tampered': False,
                'prev_reason': '',
                'movement_history': [],
                'first_movement_timestamp': None,
                'last_inserted_reason': None
            },
            'Entry Exit': {
                'tracker': {},
                'people_entering': {},
                'entering': set(),
                'people_exiting': {},
                'exiting': set(),
                'last_logged_enter': 0,
                'last_logged_exit': 0,
                'next_id': 0
            },
            'Occupancy Monitoring': {
                'count_history': deque(maxlen=20),
                'first_timestamp': None,
                'last_inserted_count': None,
                'prev_count': 0
            },
            'Employee Unavailability': {
                'frame_count': 0,
                'fps': 30,
                'current_state': 'available',
                'state_start_time': 0,
                'pending_state': None,
                'pending_state_start_time': None,
                'STATE_CHANGE_THRESHOLD': 2,
                'DISAPPEARANCE_THRESHOLD': 10
            },
            'Mobile Usage': {
                'usage_buffer': [],
                'usage_start_time': None,
                'last_usage_time': None,
                'is_usage_confirmed': False,
                'confirmation_frames': 20,
                'distance_threshold': 400,
                'prev_phone_usage': False
            },
            'Attended/Unattended Time': {
                'frame_count': 0,
                'fps': 30,
                'current_state': 'unattended',
                'state_start_time': 0,
                'pending_state': None,
                'pending_state_start_time': None,
                'STATE_CHANGE_THRESHOLD': 2,
                'DISAPPEARANCE_THRESHOLD': 10
            },
            'Customer Staff Ratio': {
                'frame_count': 0,
                'fps': 30,
                'last_db_insert_time': 0,
                'max_employee_count': 0,
                'max_customer_count': 0,
                'DB_UPDATE_INTERVAL': 10
            },
            'Dwell Time': {
                'frame_count': 0,
                'fps': 30,
                'customer_present': False,
                'dwell_start_time': 0,
                'accumulated_dwell_time': 0,
                'last_customer_time': 0,
                'no_customer_frames': 0,
                'MIN_NO_CUSTOMER_FRAMES': 25
            },
            'Fall/Slip': {
                'tracker': CentroidTracker(max_disappeared=50, max_distance=100),
                'frame_count': 0,
                'prev_fall_detected': False,
                'fps': 30,
                'frames_folder': os.path.join(settings.OUTPUT_VIDEO_DIR, "fall_frames")
            },
            'Billing Alerts': {
                'continuous_counter': 0,
                'paper_absent_counter': 0,
                'billing_done': False,
                'last_billing_state': False,
                'billing_count': 0,
                'last_alert_reason': None,
                'customer_no_bill_start': None,
                'bill_no_customer_start': None,
                'last_count_timestamp': datetime.now(),
                'fps': 30,
                'frame_count': 0
            },
            'Billing Counter': {
                'count_history': deque(maxlen=20),
                'first_timestamp': None,
                'last_inserted_count': None,
                'frame_count': 0,
                'fps': 30
            }
        }
        
        self.frame_count = 0
        self.is_processing = False
        self._loop = asyncio.get_event_loop()
        self.video_writers = {}
        self.output_dir = settings.OUTPUT_VIDEO_DIR
        self.start_time = None
        
        os.makedirs(self.output_dir, exist_ok=True)
    
    async def read_frame_async(self, cap):
        """Read frame asynchronously using ThreadPoolExecutor"""
        with ThreadPoolExecutor() as executor:
            ret, frame = await self._loop.run_in_executor(executor, cap.read)
        return ret, frame
    
    async def initialize_video_writer(self, rule_name: str, cap):
        """Initialize video writer for a specific rule"""
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = os.path.join(
                self.output_dir,
                f"{self.camera_name}_{rule_name.replace('/', '_').replace(' ', '_')}_{self.session_id}_{timestamp}.mp4"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                output_filename, fourcc, fps, (width, height)
            )
            
            if not video_writer.isOpened():
                logger.error(f"Failed to initialize video writer for {output_filename}")
                return None
                
            logger.info(f"Video writer initialized for rule {rule_name}: {output_filename}")
            return video_writer
            
        except Exception as e:
            logger.error(f"Error initializing video writer for rule {rule_name}: {e}")
            return None
    
    async def write_frame_async(self, rule_name: str, frame: np.ndarray):
        """Write frame to video file for a specific rule asynchronously"""
        if rule_name in self.video_writers and self.video_writers[rule_name]:
            try:
                with ThreadPoolExecutor() as executor:
                    await self._loop.run_in_executor(executor, self.video_writers[rule_name].write, frame)
            except Exception as e:
                logger.error(f"Error writing frame for rule {rule_name}: {e}")
    
    async def start_processing(self):
        """Start camera processing with all assigned rules"""
        self.is_processing = True
        self.start_time = datetime.now()
        logger.info(f"Starting processing for camera {self.camera_name} with rules: {self.rules}")
        
        try:
            await self._process_camera_stream()
        except asyncio.CancelledError:
            logger.info(f"Processing cancelled for camera {self.camera_name}")
            raise
        except Exception as e:
            logger.error(f"Error processing camera {self.camera_name}: {e}")
        finally:
            await self._generate_processing_report()
            self._cleanup()
    
    async def _process_camera_stream(self):
        """Main processing loop for camera stream"""
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            raise ValueError(f"Could not open video stream: {self.stream_url}")
        
        logger.info(f"Camera {self.camera_name} stream opened successfully")
        
        try:
            for rule_name in self.rules:
                if rule_name in self.rule_functions:
                    self.video_writers[rule_name] = await self.initialize_video_writer(rule_name, cap)
            
            while self.is_processing:
                ret, frame = await self.read_frame_async(cap)
                if not ret:
                    logger.warning(f"Failed to read frame from camera {self.camera_name}")
                    break
                
                self.frame_count += 1
                
                if self.frame_count % settings.FRAME_INTERVAL == 0:
                    await self._process_frame(frame)
                    
                await asyncio.sleep(0.001)
                    
        except Exception as e:
            logger.error(f"Error in camera processing loop: {e}")
        finally:
            cap.release()
    
    def _cleanup(self):
        """Clean up resources"""
        self.is_processing = False
        for rule_name, video_writer in self.video_writers.items():
            if video_writer:
                try:
                    video_writer.release()
                    logger.info(f"Video writer released for camera {self.camera_name}, rule {rule_name}")
                except Exception as e:
                    logger.error(f"Error releasing video writer for rule {rule_name}: {e}")
        self.video_writers.clear()
    
    async def _generate_processing_report(self):
        """Generate a processing report and save it to a file"""
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Generate timestamp for report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = os.path.join(
                self.output_dir,
                f"processing_report_{self.camera_name}_{timestamp}.txt"
            )
            
            # Calculate processing duration
            processing_duration = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
            
            # Calculate time complexity (simplified estimation)
            time_complexity = "O(n + m * r)"
            complexity_explanation = (
                "Time Complexity Analysis:\n"
                "- Frame processing: O(n) where n is number of pixels\n"
                "- YOLO detection: O(m) where m is number of detections\n"
                "- Rule processing: O(r) where r is number of rules\n"
                "- Combined: O(n + m * r)"
            )
            
            # Count active threads (approximation)
            thread_count = len(ThreadPoolExecutor._threads) if hasattr(ThreadPoolExecutor, '_threads') else 0
            
            # Generate report content
            report_content = f"""
Camera Processing Report
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Camera Name: {self.camera_name}
Session ID: {self.session_id}
Stream URL: {self.stream_url}

Processing Details
-----------------
Number of Rules: {len(self.rules)}
Rules Applied: {', '.join(self.rules) if self.rules else 'None'}
Processing Duration: {processing_duration:.2f} seconds

Performance Analysis
------------------
Time Complexity: {time_complexity}
{complexity_explanation}

Resource Usage
-------------
Active Threads: {thread_count}
Frame Count Processed: {self.frame_count}
Video Writers Active: {len(self.video_writers)}

Directory Information
--------------------
Output Directory: {self.output_dir}
Report Filename: {report_filename}
"""
            
            # Write report to file
            with open(report_filename, 'w') as f:
                f.write(report_content)
                
            logger.info(f"Processing report generated successfully: {report_filename}")
            return report_filename
            
        except Exception as e:
            logger.error(f"Error generating processing report: {e}")
            return None
    
    async def _process_frame(self, frame: np.ndarray):
        """Process single frame with YOLO inference and rule execution"""
        try:
            detections = self.yolo_manager.predict(frame, settings.YOLO_CONFIDENCE_THRESHOLD)
            spatial_data = self.spatial_intelligence.analyze_detections(
                detections, self.rule_states['Entry Exit']
            )
            
            for rule_name in self.rules:
                if rule_name in self.rule_functions:
                    frame_copy = frame.copy()
                    await self.rule_functions[rule_name](frame_copy, spatial_data)
                    await self.write_frame_async(rule_name, frame_copy)
                    
        except Exception as e:
            logger.error(f"Error processing frame for camera {self.camera_name}: {e}")
    
    async def _process_intrusion_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.intrusion import process_intrusion_rule
        await process_intrusion_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Intrusion']
        )
    
    async def _process_camera_tempering_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.camera_tempering import process_camera_tempering_rule
        await process_camera_tempering_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Camera Tempering']
        )
    
    async def _process_entry_exit_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.entry_exit import process_entry_exit_rule
        await process_entry_exit_rule(
            frame, spatial_data, self.camera_name, self.session_id, 
            self.mongo_credentials, self.rule_states['Entry Exit']
        )

    async def _process_occupancy_monitoring_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.occupancy_monitoring import process_occupancy_monitoring_rule
        await process_occupancy_monitoring_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Occupancy Monitoring']
        )
        
    async def _process_employee_unavailability_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.employee_unavailability import process_employee_unavailability_rule
        await process_employee_unavailability_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Employee Unavailability']
        )   
    
    async def _process_mobile_usage_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.mobile_usage import process_mobile_usage_rule
        await process_mobile_usage_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Mobile Usage']
        )
    
    async def _process_attended_unattended_time_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.attended_unattended import process_attended_unattended_time_rule
        await process_attended_unattended_time_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Attended/Unattended Time']
        )
   
    async def _process_customer_staff_ratio_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.customer_staff import process_customer_staff_ratio_rule
        await process_customer_staff_ratio_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Customer Staff Ratio']
        )

    async def _process_dwell_time_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.dwell_time import process_dwell_time_rule
        await process_dwell_time_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Dwell Time']
        )

    async def _process_fall_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.fall import process_fall_rule
        await process_fall_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Fall/Slip']
        )

    async def _process_billing_alerts_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.billing_alerts import process_billing_alerts_rule
        await process_billing_alerts_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Billing Alerts']
        )

    async def _process_billing_counter_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.billing_counter import process_billing_counter_rule
        await process_billing_counter_rule(
            frame, spatial_data, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Billing Counter']
        )

    def stop_processing(self):
        """Stop camera processing"""
        self.is_processing = False
        logger.info(f"Stopping processing for camera {self.camera_name}")