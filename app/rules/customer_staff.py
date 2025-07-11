# app/rules/customer_staff.py
import cv2 #type: ignore
import numpy as np
from datetime import datetime
from typing import Dict, Any
from app.db.database import get_collection
from app.rules.utils.black_ratio import calculate_black_ratio

def is_employee_present(frame, detections):
    """Check if an employee (person with black shirt) is present and return employee detections"""
    if len(detections) == 0:
        return False, []
    
    employee_detections = []
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.85:
            black_ratio = calculate_black_ratio(frame, x1, y1, x2, y2)
            if black_ratio > 0.2:
                print(black_ratio)
                employee_detections.append(det)
    
    # Return the results - this was missing!
    return len(employee_detections) > 0, employee_detections

async def process_customer_staff_ratio_rule(
    frame: np.ndarray,
    spatial_data: Dict[str, Any],
    camera_name: str,
    session_id: str,
    mongo_credentials: Dict[str, str],
    rule_state: Dict[str, Any]
):
    """Process Customer Staff Ratio rule to count employees and customers."""
    # MongoDB setup
    collection = get_collection("customer_staff")

    # Get FPS from rule state
    fps = rule_state.get('fps', 30)

    # Increment frame count and calculate current time
    rule_state['frame_count'] = rule_state.get('frame_count', 0) + 1
    current_time = rule_state['frame_count'] / fps

    # Initialize state if not present
    if 'last_db_insert_time' not in rule_state:
        rule_state['last_db_insert_time'] = 0
        rule_state['max_employee_count'] = 0
        rule_state['max_customer_count'] = 0
        rule_state['DB_UPDATE_INTERVAL'] = 10  # Seconds between DB updates

    # Convert spatial_data detections to format expected by is_employee_present
    formatted_detections = [
        (
            det['bbox'][0],  # x1
            det['bbox'][1],  # y1
            det['bbox'][2],  # x2
            det['bbox'][3],  # y2 (fixed: was y3)
            det['confidence'],  # score
            0 if det['class_name'] == 'person' else -1  # class_id (assuming person is class 0)
        )
        for det in spatial_data.get('detections', [])
    ]

    # Detect employees
    employee_detected, employee_detections = is_employee_present(frame, formatted_detections)

    # Identify customers (persons not classified as employees)
    customer_detections = []
    for det in formatted_detections:
        is_employee = False
        for emp_det in employee_detections:
            if np.array_equal(det, emp_det):
                is_employee = True
                break
        if not is_employee and int(det[5]) == 0 and det[4] > 0.85:
            customer_detections.append(det)

    # Count detections
    employee_count = len(employee_detections)
    customer_count = len(customer_detections)

    # Update maximum counts for the current interval
    rule_state['max_employee_count'] = max(rule_state['max_employee_count'], employee_count)
    rule_state['max_customer_count'] = max(rule_state['max_customer_count'], customer_count)

    # Store counts in MongoDB periodically
    if current_time - rule_state['last_db_insert_time'] >= rule_state['DB_UPDATE_INTERVAL']:
        try:
            collection.insert_one({
                "timestamp": datetime.now(),
                "camera_name": camera_name,
                "employee_count": rule_state['max_employee_count'],
                "customer_count": rule_state['max_customer_count'],
                "session_id": session_id
            })
            print(f"DB Update at {current_time:.2f}s: Employees = {employee_count}, Customers = {customer_count}, "
                  f"Max Employees = {rule_state['max_employee_count']}, Max Customers = {rule_state['max_customer_count']}")
            rule_state['last_db_insert_time'] = current_time
            # Reset max counts for the next interval
            rule_state['max_employee_count'] = 0
            rule_state['max_customer_count'] = 0
        except Exception as e:
            print(f"DB Error at {current_time:.2f}s: {e}")

    # Draw bounding boxes and text
    for det in employee_detections:
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.70:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            text = f"Employee: {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    for det in customer_detections:
        x1, y1, x2, y2, score, class_id = det
        if int(class_id) == 0 and score > 0.70:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            text = f"Customer: {score:.2f}"
            cv2.putText(frame, text, (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display counts
    cv2.putText(frame, f"Employees: {employee_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Customers: {customer_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Debug information
    print(f"Frame {rule_state['frame_count']}: Employees = {employee_count}, Customers = {customer_count}, "
          f"Max Employees = {rule_state['max_employee_count']}, Max Customers = {rule_state['max_customer_count']}")