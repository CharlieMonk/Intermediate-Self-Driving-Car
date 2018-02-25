import numpy as np
import cv2

rectSideCheck = lambda polygon: len(polygon)<6
lambd = lambda perimeter: perimeter*0.02
def isRect(edge):
    perimeter = cv2.arcLength(edge, True)
    polygon = cv2.approxPolyDP(edge, lambd(perimeter), True)
    return rectSideCheck(polygon)

def isRectDiagnostic(edge):
    perimeter = cv2.arcLength(edge, True)
    polygon = cv2.approxPolyDP(edge, lambd(perimeter), True)
    isRect = rectSideCheck(polygon)
    return isRect, polygon
