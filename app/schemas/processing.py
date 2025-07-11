from pydantic import BaseModel
from typing import Dict, List

class ProcessingRequest(BaseModel):
    camera_names: List[str]
    camera_rules: Dict[str, List[str]]  # camera_name -> list of rules

class ProcessingResponse(BaseModel):
    session_id: str
    message: str
    cameras: List[str]