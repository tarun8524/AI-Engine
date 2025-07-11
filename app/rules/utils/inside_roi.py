import cv2 #type: ignore
import numpy as np

def is_inside_zone(x_center, y_center, zone_points):
    """
    Check if a point lies inside a polygonal zone using OpenCV's pointPolygonTest.
    
    Args:
        x_center (float): X-coordinate of the point (e.g., center of a detection).
        y_center (float): Y-coordinate of the point (e.g., center of a detection).
        zone_points (np.ndarray): Array of shape (n, 1, 2) defining the polygon vertices.
    
    Returns:
        bool: True if the point is inside or on the boundary of the zone, False otherwise.
    """
    point = np.array([x_center, y_center], dtype=np.float32)
    return cv2.pointPolygonTest(zone_points, point, False) >= 0