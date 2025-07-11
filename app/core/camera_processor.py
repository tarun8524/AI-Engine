import cv2 #type: ignore
import numpy as np
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
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
            'Billing Counter': self._process_billing_counter_rule,
            'Shelf Occupancy Monitoring': self._process_shelf_occupancy_rule
        }

        # Rule states
        self.rule_states = {
             'Intrusion': {
                'prev_intrusion_detected': False,
                'intruded_ids': set()
            },
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
                'fps': 30
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
            },
            'Shelf Occupancy Monitoring': {
                'shelf_count': 0,
                'empty_shelf_count': 0,
                'avg_empty_percent': 0,
                'last_log_timestamp': datetime.now(),
                'SHELF_LOG_INTERVAL': timedelta(seconds=10)
            }
        }
        
        self.frame_count = 0
        self.is_processing = False
        self._loop = asyncio.get_event_loop()
        self.video_writers = {}
        self.output_dir = settings.OUTPUT_VIDEO_DIR
        
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
            # Initialize YOLO models
            await self.yolo_manager.initialize_model("yolov8m.pt")
            if "Shelf Occupancy Monitoring" in self.rules:
                await self.yolo_manager.initialize_model("shelf occupancy.pt")
            
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
            # Process Shelf Occupancy separately without spatial data
            if "Shelf Occupancy Monitoring" in self.rules:
                frame_copy = frame.copy()
                await self.rule_functions["Shelf Occupancy Monitoring"](frame_copy, {})
                await self.write_frame_async("Shelf Occupancy Monitoring", frame_copy)
            
            # Process other rules with shared YOLO model and spatial data
            detections = self.yolo_manager.predict(frame, settings.YOLO_CONFIDENCE_THRESHOLD, model_path="yolov8m.pt")
            spatial_data = self.spatial_intelligence.analyze_detections(
                detections, self.rule_states['Entry Exit']
            )
            
            for rule_name in self.rules:
                if rule_name != "Shelf Occupancy Monitoring" and rule_name in self.rule_functions:
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

    async def _process_shelf_occupancy_rule(self, frame: np.ndarray, spatial_data: Dict[str, Any]):
        from app.rules.shelf_monitoring import process_shelf_occupancy_rule
        await process_shelf_occupancy_rule(
            frame, self.camera_name, self.session_id,
            self.mongo_credentials, self.rule_states['Shelf Occupancy Monitoring'],
            self.yolo_manager
        )

    def stop_processing(self):
        """Stop camera processing"""
        self.is_processing = False
        logger.info(f"Stopping processing for camera {self.camera_name}")