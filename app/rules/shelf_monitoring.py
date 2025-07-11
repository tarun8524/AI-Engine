import cv2 #type: ignore
import numpy as np
from datetime import datetime, timedelta
from app.db.database import get_collection
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

def get_centroid(x1, y1, x2, y2):
    return ((x1 + x2) // 2, (y1 + y2) // 2)

async def process_shelf_occupancy_rule(frame: np.ndarray, camera_name: str, session_id: str, 
                                      mongo_credentials: dict, rule_state: dict, yolo_manager):
    """Process shelf occupancy rule using custom YOLO model"""
    try:
        shelf_collection = get_collection("shelf_occupancy")
        logger.info("MongoDB connection established for shelf occupancy")
        
        # Run YOLO detection with custom model
        results = yolo_manager.predict(frame, conf_threshold=0.65, model_path=settings.MODEL_PATH_SHELF_OCCUPANCY)
        
        # Lists to store shelf and empty boxes
        shelf_boxes = []
        empty_boxes = []
        individual_empty_percents = []
        
        # Process detection results
        for detection in results:
            x1, y1, x2, y2 = detection["bbox"]
            label = detection["class_name"].lower()
            conf = detection["confidence"]
            area = (x2 - x1) * (y2 - y1)
            centroid = detection["centroid"]
            
            if "shelf" in label:
                shelf_boxes.append({"bbox": (x1, y1, x2, y2), "area": area, "centroid": centroid})
            elif "empty" in label:
                empty_boxes.append({"bbox": (x1, y1, x2, y2), "area": area, "centroid": centroid})
        
        # Calculate empty space percentage for each empty box
        for empty in empty_boxes:
            x1, y1, x2, y2 = empty["bbox"]
            empty_c = empty["centroid"]
            
            # Match to nearest shelf
            min_dist = float("inf")
            matched_shelf = None
            for shelf in shelf_boxes:
                shelf_c = shelf["centroid"]
                dist = np.linalg.norm(np.array(empty_c) - np.array(shelf_c))
                if dist < min_dist:
                    min_dist = dist
                    matched_shelf = shelf
            
            # Calculate % of empty space in matched shelf
            empty_percent = 0
            if matched_shelf and matched_shelf["area"] > 0:
                empty_percent = (empty["area"] / matched_shelf["area"]) * 100
                individual_empty_percents.append(empty_percent)
            
            # Draw bounding box and % text
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"empty {empty_percent:.1f}%", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Draw shelf bounding boxes
        for shelf in shelf_boxes:
            x1, y1, x2, y2 = shelf["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "shelf", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Calculate summary metrics
        shelf_count = len(shelf_boxes)
        empty_shelf_count = len(empty_boxes)
        avg_empty_percent = sum(individual_empty_percents) / len(individual_empty_percents) if individual_empty_percents else 0
        
        # Update rule state
        rule_state["shelf_count"] = shelf_count
        rule_state["empty_shelf_count"] = empty_shelf_count
        rule_state["avg_empty_percent"] = avg_empty_percent
        
        # Draw global summary box
        cv2.rectangle(frame, (0, 0), (250, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"Shelves: {shelf_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Filled shelfs: {max(shelf_count - empty_shelf_count, 0)}", (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Unfilled shelfs: {empty_shelf_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Avg empty %: {avg_empty_percent:.1f}%", (10, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Save to MongoDB every 10 seconds if all values are greater than 0
        current_time = datetime.now()
        if current_time - rule_state["last_log_timestamp"] >= rule_state["SHELF_LOG_INTERVAL"]:
            if shelf_count > 0 and empty_shelf_count > 0 and avg_empty_percent > 0:
                data = {
                    "timestamp": current_time,
                    "camera_name": camera_name,
                    "shelves": shelf_count,
                    "empty_shelves": empty_shelf_count,
                    "avg_empty_space": round(avg_empty_percent, 1)
                }
                try:
                    shelf_collection.insert_one(data)
                    logger.info(f"Inserted shelf occupancy to DB: {data} at {current_time}")
                except Exception as e:
                    logger.error(f"Error inserting shelf occupancy to DB: {str(e)}")
                rule_state["last_log_timestamp"] = current_time
            else:
                logger.info(f"Skipped MongoDB insertion: shelf_count={shelf_count}, "
                           f"empty_shelf_count={empty_shelf_count}, avg_empty_percent={avg_empty_percent}")
        
        return frame
        
    except Exception as e:
        logger.error(f"Error in shelf occupancy processing: {e}")
        return frame