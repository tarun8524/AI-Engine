from pydantic_settings import BaseSettings #type: ignore
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
        {"id": 3, "rule": "Entry Exit"},
        {"id": 4, "rule": "Employee Unavailability"},
        {"id": 5, "rule": "Occupancy Monitoring"},
        {"id": 6, "rule": "Mobile Usage"},
        {"id": 7, "rule": "Customer Staff Ratio"},
        {"id": 8, "rule": "Billing Counter"},
        {"id": 9, "rule": "Dwell Time"},
        {"id": 10, "rule": "Billing Alerts"},
        {"id": 11, "rule": "Fall/Slip"},
        {"id": 12, "rule": "Attended/Unattended Time"},
        {"id": 13, "rule": "Shelf Occupancy Monitoring"},
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