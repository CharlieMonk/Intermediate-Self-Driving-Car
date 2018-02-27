import numpy as np
import cv2

class ShapeDetection:
    def __init__(self, contour):
        self.contour = contour
    rectSideCheck = lambda polygon: len(polygon)<6
    dist = lambda self, pt1, pt2: ((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)**0.5
    slope = lambda self, pt1, pt2: (pt2[1]-pt1[1])/(pt2[0]-pt1[0])
    def findRect(self):
        rect = cv2.minAreaRect(self.contour)
        box = np.int0(cv2.boxPoints(rect))
        return box
    def isRect(self):
        perimeter = cv2.arcLength(self.contour, True)
        polygon = cv2.approxPolyDP(self.contour, perimeter*0.02, True)
        box = self.findRect()
        ratio = self.dist(box[0], box[1])/self.dist(box[1], box[2])
        slope = abs(self.slope(box[1],box[2]))
        return (ratio>3 or 1/ratio>3) and (slope>1)
    def isRectDiagnostic(self):
        perimeter = cv2.arcLength(self.contour, True)
        polygon = cv2.approxPolyDP(self.contour, perimeter*0.02, True)
        isRect = rectSideCheck(polygon)
        return isRect, polygon
