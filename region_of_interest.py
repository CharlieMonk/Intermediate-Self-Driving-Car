import numpy as np
import cv2

def getVerticies(shape):
    width, height = shape
    bottom_left = (height, 0)
    bottom_right = (height, width)
    upper_left = (0.25*height, 0.8*width)
    upper_right = (0.25*height, 0.2*width)
    verticies = np.array([[bottom_left, upper_left, bottom_right, upper_right]])
    return np.int32(verticies)

def roi(edges, img):
    mask = np.zeros_like(edges)
    verticies = getVerticies(edges.shape)
    print("Shape:", edges.shape,img.shape)
    print("Verticies:", verticies)
    cv2.fillPoly(mask, verticies, (255,255))
    img	= cv2.polylines(img, verticies, True, (255,0,0))
    return cv2.bitwise_and(edges, mask)
