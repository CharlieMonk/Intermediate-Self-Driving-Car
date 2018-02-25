import numpy as np
import cv2
import shape_detection as detectShape

def findContours(edges):
    # Finds the contours of Canny edges and computes the solidity
    for edge in edges:
        img2, contour, hierarchy = cv2.findContours(edge, 1, 2)
        contour_area = 1# cv2.contourArea(contour)
        if(contour_area > 50):
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = float(contour_area)/hull_area

def drawBoundingRects(img, edges):
    # Finds the contours of the Canny edges
    img2, contours, hierarchy = cv2.findContours(edges, 1, 2)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if(contour_area > 20):
            print("Contour area: " + str(contour_area))
            if(detectShape.isRect(contour)):
                rect = cv2.minAreaRect(contour)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(img, [box], 0, (0,0,255))
            else:
                isRect, polygon = detectShape.isRectDiagnostic(contour)
                color = (255,0,0)#*isRect + (0,0,0)*(not isRect)
                box = cv2.polylines(img, [polygon], True, color)
            # # Finds the portion of the contour that is convex
            # hull = cv2.convexHull(contour)
            # hull_area = cv2.contourArea(hull)
            # # Finds how much of the contour is occupied by a convex shape
            # solidity = float(contour_area)/hull_area
            # if solidity>0.05:
            #     rect = cv2.minAreaRect(contour)
            #     box = np.int0(cv2.boxPoints(rect))
            #     cv2.drawContours(img, [box], 0, (0,0,255))
            #     print(solidity)
    return img2

def findRoadLines(image_path):
    img = cv2.imread(image_path, 1)

    # Lower and upper ranges for yellow center lines and white lane lines
    center_lines_lower = np.array([0, 104, 210])
    center_lines_upper = np.array([255, 255, 255]) #([160, 234, 255])

    # Mask image using center_lines_lower and center_lines_upper
    res = cv2.inRange(img, center_lines_lower, center_lines_upper)

    # Morph closing on image
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)

    # Lower and upper canny thresholds
    lower_canny = 100
    upper_canny = 200

    # Use canny to detect edges
    edges = cv2.Canny(closed, lower_canny, upper_canny)
    # Fit lines to the canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

    # If thresholding didn't work, use the original image
    if (lines == None):
        edges = cv2.Canny(img, lower_canny, upper_canny)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
    #findContours(edges)
    # Draw the lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=7)

    # Draw rotated rectangles around the contours
    img2 = drawBoundingRects(img, edges)

    cv2.imshow("edges", img2)
    return img

# Run findRoadLines on a test image
linesFound = findRoadLines("/Users/cbmonk/AnacondaProjects/Advanced-Self-Driving-Car/TestImages/15.png")
cv2.imshow("Lines detected", linesFound)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
