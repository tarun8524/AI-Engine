from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2 #type: ignore
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
        await shared_yolo_manager.initialize_model(settings.YOLO_MODEL_PATH_8m)
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

# @router.get("/sessions")
# async def get_active_sessions():
#     """Get all active processing sessions with their status"""
#     global active_tasks, task_status
    
#     sessions = {}
#     for session_id, task in active_tasks.items():
#         status_info = task_status.get(session_id, {
#             "camera_name": "unknown",
#             "status": "running",
#             "message": "Processing running"
#         })
        
#         sessions[session_id] = {
#             "camera": status_info["camera_name"],
#             "status": "running" if not task.done() else "completed",
#             "message": status_info["message"]
#         }
    
#     return sessions

# @router.get("/session-results/{session_id}")
# async def get_session_results(session_id: str):
#     """Get results for a specific session"""
#     global task_status
    
#     if session_id not in task_status:
#         raise HTTPException(status_code=404, detail="Session not found")
    
#     return {
#         "session_id": session_id,
#         "results": [task_status[session_id]]
#     }

async def cleanup_async_tasks():
    """Cleanup all async tasks on application shutdown"""
    logger.info("Cleaning up all async tasks")
    await stop_all_processing()