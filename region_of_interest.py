import numpy as np
import cv2

def getVerticies(shape):
    height, width = shape
    bottom_l = (0, height)
    mid_l = (0, bottom_l[1]-150)
    up_l = (width/4, 0.4*height)
    up_r = (0.75*width, 0.4*height)
    bottom_r = (width, height)
    mid_r = (bottom_r[0], bottom_r[1]-150)
    verticies = np.array([[bottom_l, mid_l, up_l, up_r, mid_r, bottom_r]])
    return np.int32(verticies)

def roi(edges):
    mask = np.zeros_like(edges)
    verticies = getVerticies(edges.shape)
    cv2.fillPoly(mask, verticies, 255)
    edges = cv2.bitwise_and(mask, edges)
    return edges
