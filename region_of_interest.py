import numpy as np
import cv2

def getVerticies(shape):
    height, width = shape
    bottom_left = (10, height-10)
    upper_left = (width/4, 0.4*height)
    upper_right = (0.75*width, 0.4*height)
    bottom_right = (width-10, height-10)
    verticies = np.array([[bottom_left, upper_left, upper_right, bottom_right]])
    return np.int32(verticies)

def roi(edges, img):
    mask = np.zeros_like(edges)
    verticies = getVerticies(edges.shape)
    cv2.fillPoly(mask, verticies, (255,255,255))
    img	= cv2.polylines(img, verticies, True, (255,0,0))
    return cv2.bitwise_and(edges, mask)
