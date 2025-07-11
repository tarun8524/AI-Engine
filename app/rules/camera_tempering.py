import cv2 #type: ignore
import numpy as np
from datetime import datetime
from typing import Dict, Any
from collections import deque
from app.db.database import get_collection

def detect_occlusion(frame, brightness_threshold=40):
    """Detect camera occlusion based on brightness"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    if avg_brightness < brightness_threshold:
        return True
    return False

def detect_movement(prev_frame, curr_frame, diff_threshold=10000):
    """Detect camera movement based on frame difference"""
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    score = np.sum(gray_diff)//10000
    print("score:", score)
    if score > diff_threshold:
        return True
    return False

async def process_camera_tempering_rule(frame: np.ndarray, spatial_data: Dict[str, Any], 
                                      camera_name: str, session_id: str, 
                                      mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process camera tempering detection rule"""
    
    # MongoDB setup
    collection = get_collection("camera_tempering")
    
    # Get previous frame from rule state if available
    prev_frame = rule_state.get('prev_frame', None)
    
    # Initialize all required keys if not exists
    if 'movement_history' not in rule_state:
        rule_state['movement_history'] = deque(maxlen=5)
    
    if 'first_movement_timestamp' not in rule_state:
        rule_state['first_movement_timestamp'] = None
    
    if 'last_inserted_reason' not in rule_state:
        rule_state['last_inserted_reason'] = None
    
    current_tampered = False
    current_reason = ""
    
    # Check for occlusion
    if detect_occlusion(frame):
        current_tampered = True
        current_reason = "occlusion"
        print("[ALERT] Possible camera occlusion detected!")
    elif prev_frame is not None:
        # Check for movement
        movement_detected = detect_movement(prev_frame, frame)
        rule_state['movement_history'].append(movement_detected)
        
        # Track first timestamp for movement detection
        if movement_detected and (not rule_state['movement_history'] or 
                                not rule_state['movement_history'][-2] if len(rule_state['movement_history']) >= 2 else True):
            rule_state['first_movement_timestamp'] = datetime.now()
        
        # Check if movement detected in at least 2 of last 5 frames
        if len(rule_state['movement_history']) >= 5 and sum(rule_state['movement_history']) >= 2:
            current_tampered = True
            current_reason = "camera movement"
            print("[ALERT] Possible camera movement detected!")
    
    # Save to DB only if:
    # - Previous frame was not tampered and current frame is tampered, or
    # - Current frame is tampered and reason has changed
    if ((not rule_state['prev_tampered'] and current_tampered) or 
        (current_tampered and current_reason != rule_state['prev_reason'])):
        
        cv2.putText(frame, f"TAMPERING DETECTED: {current_reason.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Use first_movement_timestamp for movement, otherwise current timestamp
        timestamp = (rule_state['first_movement_timestamp'] if current_reason == "camera movement" 
                    else datetime.now())
        
        # Only insert if reason is different from last inserted
        if current_reason != rule_state['last_inserted_reason']:
            data = {
                "timestamp": timestamp,
                "camera_name": camera_name,
                "cam_temp": True,
                "reason": current_reason
            }
            try:
                collection.insert_one(data)
                rule_state['last_inserted_reason'] = current_reason
            except Exception as e:
                print(f"Error inserting to DB: {e}")
    
    # Update rule state
    rule_state['prev_tampered'] = current_tampered
    rule_state['prev_reason'] = current_reason
    rule_state['prev_frame'] = frame.copy()