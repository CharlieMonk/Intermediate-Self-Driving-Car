import numpy as np
import cv2
from shape_detection import ShapeDetection
import time
import region_of_interest as ROI

def drawObstacles(img, bgr_img, edges):
    # Finds the contours of the Canny edges
    img2, contours, hierarchy = cv2.findContours(edges, 1, 2)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if(contour_area > 10 and contour_area<9000):
            rect = ShapeDetection(contour, bgr_img.shape)
            if(rect.isRect()):
                pt = rect.findRectAndPt()[1]
                cv2.circle(bgr_img, pt, 3, (0,0,255))
                #cv2.drawContours(bgr_img, [box], 0, (0,0,255))
            diagnosticOn = False
            # If diagnostic mode is on, show the non-rectangular contours
            if(diagnosticOn):
                isRect, polygon = isRectDiagnostic(contour)
                color = (255,0,0)#*isRect + (0,0,0)*(not isRect)
                box = cv2.polylines(bgr_img, [polygon], True, color)
    return img2

def findRoadLines(image_path):
    bgr_img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
    img_find_noise = bgr_img

    # Lower and upper canny thresholds
    lower_canny = 100
    upper_canny = 300

    # Use the BGR image to find obstacles because HLS has too much noise
    edges2 = cv2.Canny(img_find_noise, lower_canny, upper_canny)
    # Draw the obstacles
    img2 = drawObstacles(img, bgr_img, edges2)

    # Use canny on HLS to detect edges
    #edges1 = cv2.Canny(closed, lower_canny, upper_canny)
    edges_ = cv2.Canny(img, lower_canny, upper_canny)
    # Isolate region of interest
    edges = ROI.roi(edges_, bgr_img)
    # Fit lines to Canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 130, minLineLength=100, maxLineGap=7)
    # Draw the lines
    if(lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Don't draw lines that are too slanted (to eliminate noise)
            if(abs((y2-y1)/(x2-x1))>0.5):
                cv2.line(bgr_img, (x1, y1), (x2, y2), (0,255,0), thickness=7)

    cv2.imshow("edges", edges)
    return img, bgr_img

time0 = time.time()
# Run findRoadLines on a test image
img, bgr_img = findRoadLines("/Users/cbmonk/AnacondaProjects/Advanced-Self-Driving-Car/TestImages/26.png")
cv2.imshow("HSV", img)
cv2.imshow("BGR", bgr_img)
print("Total time:", time.time()-time0)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
