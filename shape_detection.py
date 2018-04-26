import numpy as np
import cv2

class ShapeDetection:
    def __init__(self, contour, shape):
        self.contour = contour
        self.height, self.width = shape[:2]
    rectSideCheck = lambda polygon: len(polygon)<6
    # Mini- functions on 2 points (distance between, slope of, mid- pt of)
    dist = lambda self, pt1, pt2: ((pt2[1]-pt1[1])**2 + (pt2[0]-pt1[0])**2)**0.5
    slope = lambda self, pt1, pt2: (pt2[1]-pt1[1])/(pt2[0]-pt1[0]+0.0001)
    mid = lambda self, pt1, pt2:(int((pt2[0]+pt1[0])/2), int((pt2[1]+pt1[1])/2))

    # Find the rectangle of a contour and the centroid thereof
    def findRectAndPt(self):
        rect = cv2.minAreaRect(self.contour)
        box = np.int0(cv2.boxPoints(rect))
        pt = (-10,-10)
        if(len(box)>0):
            pt = self.mid(self.mid(box[0], box[1]), self.mid(box[1],box[2]))
            #print(self.contour.shape)
            if(pt[1] > 0.75*self.height):
                pt = (-10, -10)
        return box, pt
    # Check if a rectangle is an actual rectangle (NOT noise)
    def isRect(self):
        perimeter = cv2.arcLength(self.contour, True)
        polygon = cv2.approxPolyDP(self.contour, perimeter*0.01, True)
        length = len(polygon)
        box = self.findRectAndPt()[0]
        ratio = self.dist(box[0], box[1])/self.dist(box[1], box[2])
        slope = abs(self.slope(box[1],box[2]))
        return not((ratio<6 or 1/(ratio+0.001)<6) and (slope<0.1)) and length>15
    def isRectDiagnostic(self):
        perimeter = cv2.arcLength(self.contour, True)
        polygon = cv2.approxPolyDP(self.contour, perimeter*0.01, True)
        isRect = rectSideCheck(polygon)
        return isRect, polygon
