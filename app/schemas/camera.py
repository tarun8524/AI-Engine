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
