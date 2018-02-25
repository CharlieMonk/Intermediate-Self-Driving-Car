import numpy as np
import cv2

def isRect(edge):
    perimeter = cv2.arcLength(edge, True)
    polygon = cv2.approxPolyDP(edge, perimeter*0.04, True)
    return len(polygon)>3 and len(polygon)<6
