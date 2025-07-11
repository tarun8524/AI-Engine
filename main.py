from fastapi import FastAPI
from app.api.v1.endpoints import api_router
from app.core.config import settings
from app.core.yolo_manager import YOLOManager
from app.db.database import initialize_database, db_manager
import logging
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define lifespan event
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown event handler using lifespan"""
    try:
        # --- Startup logic ---
        logger.info("Initializing database connection...")
        await initialize_database(
            connection_string=settings.MONGO_CREDENTIALS["connection_string"],
            password=settings.MONGO_CREDENTIALS["password"],
            db_name=settings.MONGO_CREDENTIALS["db_name"]
        )
        logger.info("Database connection initialized successfully")

        logger.info("Initializing YOLO model...")
        yolo_manager = YOLOManager()
        await yolo_manager.initialize_model(settings.YOLO_MODEL_PATH_8m)
        app.state.yolo_manager = yolo_manager
        logger.info("YOLO model initialized successfully")

        logger.info("Application startup completed successfully")

        yield  # --- Control is returned to the application here

    finally:
        # --- Shutdown logic ---
        try:
            db_manager.close()
            logger.info("Database connection closed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan
)

app.include_router(api_router, prefix="/api/v1")

def main():
    """Run the FastAPI application"""
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)

if __name__ == "__main__":
    main()
