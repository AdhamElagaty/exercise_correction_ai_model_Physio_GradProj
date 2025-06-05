import numpy as np
from typing import List, Tuple, Union

Coordinate = Union[List[float], Tuple[float, float], np.ndarray]

class GeometryUtils:
    @staticmethod
    def calculate_angle(a: Coordinate, b: Coordinate, c: Coordinate) -> float:
        a_np = np.array(a)
        b_np = np.array(b)
        c_np = np.array(c)

        if np.isnan(a_np).any() or np.isnan(b_np).any() or np.isnan(c_np).any():
            return np.nan
        
        ba = a_np - b_np
        bc = c_np - b_np
        
        radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360.0 - angle
            
        return angle

    @staticmethod
    def distance_point_to_line(point: Coordinate, line_p1: Coordinate, line_p2: Coordinate) -> float:
        point_np = np.array(point)
        line_p1_np = np.array(line_p1)
        line_p2_np = np.array(line_p2)

        if np.isnan(point_np).any() or np.isnan(line_p1_np).any() or np.isnan(line_p2_np).any():
            return np.nan
        
        line_vec = line_p2_np - line_p1_np
        point_vec = point_np - line_p1_np
        
        line_len_sq = np.dot(line_vec, line_vec)
        
        if line_len_sq == 0:
            return np.linalg.norm(point_np - line_p1_np)
            
        t = np.dot(point_vec, line_vec) / line_len_sq

        if t < 0.0:
            closest_point = line_p1_np
        elif t > 1.0:
            closest_point = line_p2_np
        else:
            closest_point = line_p1_np + t * line_vec
            
        return np.linalg.norm(point_np - closest_point)

    @staticmethod
    def calculate_distance(p1: Coordinate, p2: Coordinate) -> float:
        p1_np = np.array(p1)
        p2_np = np.array(p2)
        if np.isnan(p1_np).any() or np.isnan(p2_np).any():
            return np.nan
        return np.linalg.norm(p1_np - p2_np)