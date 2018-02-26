import numpy as np
import cv2
import shape_detection as detectShape
import time
import region_of_interest as ROI

def drawBoundingRects(img, bgr_img, edges):
    # Finds the contours of the Canny edges
    img2, contours, hierarchy = cv2.findContours(edges, 1, 2)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if(contour_area > 10):
            if(detectShape.isRect(contour)):
                rect = cv2.minAreaRect(contour)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(bgr_img, [box], 0, (0,0,255))
            diagnosticOn = False
            # If diagnostic mode is on, show the non-rectangular contours
            if(diagnosticOn):
                isRect, polygon = detectShape.isRectDiagnostic(contour)
                color = (255,0,0)#*isRect + (0,0,0)*(not isRect)
                box = cv2.polylines(bgr_img, [polygon], True, color)


    return img2

def findRoadLines(image_path):
    bgr_img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)

    # Lower and upper ranges for yellow center lines and white lane lines
    center_lines_lower = np.array([255, 255, 255])
    center_lines_upper = np.array([255, 255, 255]) #([160, 234, 255])

    # Mask image using center_lines_lower and center_lines_upper
    res = cv2.inRange(img, center_lines_lower, center_lines_upper)

    # Morph closing on image
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    # Lower and upper canny thresholds
    lower_canny = 100
    upper_canny = 400

    # Use canny to detect edges
    edges1 = cv2.Canny(closed, lower_canny, upper_canny)
    edges = ROI.roi(edges1, bgr_img)
    # Fit lines to the canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100,maxLineGap=10)

    # If thresholding didn't work, use the original image
    if (lines == None):
        edges = cv2.Canny(img, lower_canny, upper_canny)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    # Draw the lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(bgr_img, (x1, y1), (x2, y2), (0, 255, 0), thickness=7)

    # Draw rotated rectangles around the contours
    img2 = drawBoundingRects(img, bgr_img, edges)

    cv2.imshow("edges", img2)
    return img, bgr_img

time0 = time.time()
# Run findRoadLines on a test image
img, bgr_img = findRoadLines("/Users/cbmonk/AnacondaProjects/Advanced-Self-Driving-Car/TestImages/23.png")
cv2.imshow("HSV", img)
cv2.imshow("BGR", bgr_img)
print("Total time:", time.time()-time0)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
