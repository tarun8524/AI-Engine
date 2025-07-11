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