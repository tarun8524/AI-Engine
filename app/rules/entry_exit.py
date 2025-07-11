import cv2 #type: ignore
import numpy as np
import torch
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.tracker import Tracker

# Area definitions (from original code)
area1 =  [(1072, 1687), (1179, 1669), (1890, 2002), (1790, 2032)]# Entry area
area2 =  [(990, 1702), (1065, 1690), (1738, 2045), (1631, 2073)] # Exit area

def convert_detections_to_bbox_list(detections):
    """Convert spatial_data person detections to bbox list for tracker"""
    bbox_list = []
    for detection in detections:
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        bbox_list.append([x1, y1, x2, y2])
    return bbox_list

async def process_entry_exit_rule(frame: np.ndarray, spatial_data: Dict[str, Any], 
                                camera_name: str, session_id: str, 
                                mongo_credentials: Dict[str, str], rule_state: Dict[str, Any]):
    """Process entry/exit counting rule"""
    
    # MongoDB setup
    collection = get_collection("entry_exit")
    
    # Initialize tracker if not exists
    if 'tracker_obj' not in rule_state:
        rule_state['tracker_obj'] = Tracker()
    
    font = cv2.FONT_HERSHEY_COMPLEX
    entry_area = area1
    exit_area = area2
    
    # Process person detections
    person_detections = spatial_data.get('person_detections', [])
    
    if person_detections:
        # Convert detections to bbox list for tracker
        bbox_list = convert_detections_to_bbox_list(person_detections)
        
        # Update tracker
        bbox_id = rule_state['tracker_obj'].update(bbox_list)
        
        # Process each tracked person
        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            
            # Process person entering (going from exit_area to entry_area)
            results = cv2.pointPolygonTest(np.array(exit_area, np.int32), ((x4, y4)), False)
            if results >= 0:
                rule_state['people_entering'][id] = (x4, y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                
            if id in rule_state['people_entering']:
                results1 = cv2.pointPolygonTest(np.array(entry_area, np.int32), ((x4, y4)), False)
                if results1 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                    cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                    cv2.putText(frame, "enter", (x3, y3-10), font, (0.5), (255, 255, 255), 1)
                    cv2.putText(frame, f"ID: {id}", (x3+65, y3-10), font, (0.5), (255, 0, 255), 1)
                    rule_state['entering'].add(id)
            
            # Process person exiting (going from entry_area to exit_area)
            results2 = cv2.pointPolygonTest(np.array(entry_area, np.int32), ((x4, y4)), False)
            if results2 >= 0:
                rule_state['people_exiting'][id] = (x4, y4)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                
            if id in rule_state['people_exiting']:
                results3 = cv2.pointPolygonTest(np.array(exit_area, np.int32), ((x4, y4)), False)
                if results3 >= 0:
                    cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                    cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                    cv2.putText(frame, "exit", (x3, y3-10), font, (0.5), (255, 255, 255), 1)
                    cv2.putText(frame, f"ID: {id}", (x3+55, y3-10), font, (0.5), (255, 0, 255), 1)
                    rule_state['exiting'].add(id)
    
    # Draw regions
    cv2.polylines(frame, [np.array(entry_area, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, "1", (504, 471), font, 1, (0, 0, 0), 2)
    
    cv2.polylines(frame, [np.array(exit_area, np.int32)], True, (255, 0, 0), 2)
    cv2.putText(frame, "2", (466, 485), font, 1, (0, 0, 0), 2)
    
    cv2.putText(frame, f"Number of entering people: {len(rule_state['entering'])}", (20, 44), font, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Number of exiting people: {len(rule_state['exiting'])}", (20, 82), font, 1, (0, 0, 255), 2)
    
    # Insert data to DB if counts change
    entry_count = len(rule_state['entering'])
    exit_count = len(rule_state['exiting'])
    
    if entry_count != rule_state['last_logged_enter'] or exit_count != rule_state['last_logged_exit']:
        current_entry = entry_count - rule_state['last_logged_enter']
        current_exit = exit_count - rule_state['last_logged_exit']
        
        try:
            collection.insert_one({
                "camera_name": camera_name,
                "timestamp": datetime.now(),
                "entry_count": current_entry,
                "exit_count": current_exit
            })
        except Exception as e:
            print(f"Error inserting to MongoDB: {e}")
            
        rule_state['last_logged_enter'] = entry_count
        rule_state['last_logged_exit'] = exit_count