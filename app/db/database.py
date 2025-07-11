# app/db/database.py
from pymongo import MongoClient #type: ignore
from pymongo.errors import ConnectionFailure, OperationFailure #type: ignore
from urllib.parse import quote_plus
from typing import Tuple, Optional, Dict, Any
import logging
from contextlib import asynccontextmanager
import asyncio

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Singleton database manager for MongoDB connections"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.client: Optional[MongoClient] = None
            self.db = None
            self.credentials: Optional[Dict[str, str]] = None
            self.initialized = False
    
    async def initialize(self, connection_string: str, password: str, db_name: str):
        """Initialize database connection"""
        async with self._lock:
            if self.initialized:
                return
            
            try:
                # Ensure password is URL-encoded
                encoded_password = quote_plus(password)
                
                # Replace placeholder with the actual password
                mongo_uri = connection_string.replace("<db_password>", encoded_password)
                
                # Create client with proper connection settings
                self.client = MongoClient(
                    mongo_uri,
                    serverSelectionTimeoutMS=5000,  # 5 second timeout
                    connectTimeoutMS=10000,  # 10 second connection timeout
                    maxPoolSize=10,  # Connection pool size
                    retryWrites=True
                )
                
                # Test connection
                self.client.admin.command("ping")
                logger.info("MongoDB connection successful")
                
                # Set database
                self.db = self.client[db_name]
                self.credentials = {
                    "connection_string": connection_string,
                    "password": password,
                    "db_name": db_name
                }
                self.initialized = True
                
            except OperationFailure as auth_error:
                logger.error(f"MongoDB authentication failed: {auth_error}")
                raise Exception("Authentication failed! Check your username and password.")
            except ConnectionFailure as conn_error:
                logger.error(f"MongoDB connection failed: {conn_error}")
                raise Exception("Failed to connect to MongoDB. Check your network or MongoDB URI.")
            except Exception as e:
                logger.error(f"MongoDB initialization error: {e}")
                raise Exception(f"Database initialization error: {e}")
    
    def get_collection(self, collection_name: str):
        """Get a collection from the database"""
        if not self.initialized or self.db is None:
            raise Exception("Database not initialized. Call initialize() first.")
        return self.db[collection_name]
    
    def get_client_and_db(self) -> Tuple[MongoClient, Any]:
        """Get client and database objects"""
        if not self.initialized or self.client is None or self.db is None:
            raise Exception("Database not initialized. Call initialize() first.")
        return self.client, self.db
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None
            self.initialized = False

# Global database manager instance
db_manager = DatabaseManager()

# Convenience functions for backward compatibility
async def initialize_database(connection_string: str, password: str, db_name: str):
    """Initialize database connection"""
    await db_manager.initialize(connection_string, password, db_name)

def get_collection(collection_name: str):
    """Get a collection from the database"""
    return db_manager.get_collection(collection_name)

def get_client_and_db() -> Tuple[MongoClient, Any]:
    """Get client and database objects"""
    return db_manager.get_client_and_db()

def get_collection_cam_details():
    """Return the 'cam_details' collection if initialized."""
    return get_collection("cam_details")

def save_cam(camera_name: str, module_name: list,location: str, stream_url: str):
    """Save camera details to the 'cam_details' collection."""
    try:
        collection = get_collection_cam_details()
        camera_data = {
            "camera_name": camera_name,
            "module_names": module_name,
            "location": location,
            "stream_url": stream_url
        }
        result = collection.insert_one(camera_data)
        logger.info(f"Camera {camera_name} saved successfully")
        return result.inserted_id
    except OperationFailure as e:
        logger.error(f"Failed to save camera details: {str(e)}")
        raise Exception(f"Failed to save camera details: {str(e)}")
    except Exception as e:
        logger.error(f"Error saving camera details: {str(e)}")
        raise Exception(f"Error saving camera details: {str(e)}")

# Legacy functions for backward compatibility (deprecated)
def get_mongo_client(connection_string: str, password: str, db_name: str) -> Tuple[MongoClient, Any]:
    """Legacy function - use get_client_and_db() instead"""
    logger.warning("get_mongo_client is deprecated. Database should be initialized once at startup.")
    return get_client_and_db()