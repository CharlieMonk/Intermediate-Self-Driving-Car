import numpy as np
import cv2

rectSideCheck = lambda polygon: len(polygon)<6
def isRect(edge):
    perimeter = cv2.arcLength(edge, True)
    polygon = cv2.approxPolyDP(edge, perimeter*0.04, True)
    return rectSideCheck(polygon)

def isRectDiagnostic(edge):
    perimeter = cv2.arcLength(edge, True)
    polygon = cv2.approxPolyDP(edge, perimeter*0.04, True)
    isRect = rectSideCheck(polygon)
    return isRect, polygon
