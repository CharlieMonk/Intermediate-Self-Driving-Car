import numpy as np
import cv2
from shape_detection import ShapeDetection
import time
import region_of_interest as ROI

def drawObstacles(hsv_img, bgr_img, edges):
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

def findRoadLines(img, lower_canny=50, upper_canny=350):
    # Use canny on HLS to detect hsv_edges
    edges_ = cv2.Canny(img, lower_canny, upper_canny)
    # Isolate region of interest
    hsv_edges = ROI.roi(edges_)
    # Fit lines to Canny hsv_edges
    lines = cv2.HoughLinesP(hsv_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=7)
    return lines, hsv_edges

def analyzeImage(image_path):
    bgr_img = cv2.imread(image_path, 1)
    hsv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)
    hls_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HLS)
    img_find_noise = bgr_img

    # Lower and upper canny thresholds
    lower_canny = 150
    upper_canny = 300

    # Use the BGR image to find obstacles because HLS has too much noise
    bgr_edges = cv2.Canny(img_find_noise, lower_canny, upper_canny)
    # Draw the obstacles
    drawObstacles(hsv_img, bgr_img, bgr_edges)

    # # Use canny on HLS to detect hsv_edges
    # edges_ = cv2.Canny(hsv_img, lower_canny, upper_canny)
    # # Isolate region of interest
    # hsv_edges = ROI.roi(edges_, bgr_img)
    # # Fit lines to Canny hsv_edges
    # lines = cv2.HoughLinesP(hsv_edges, 1, np.pi/180, 130, minLineLength=100, maxLineGap=7)
    hsv_lines, hsv_edges = findRoadLines(hsv_img)#, lower_canny, upper_canny)
    hls_lines, hls_edges = findRoadLines(hls_img)#, lower_canny, upper_canny)
    lines = np.vstack((hsv_lines, hls_lines))
    # Draw the lines
    if(lines is not None):
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Don't draw lines that are too slanted (to eliminate noise)
            if(abs((y2-y1)/(x2-x1))>0.7):
                cv2.line(bgr_img, (x1, y1), (x2, y2), (0,255,0), thickness=7)

    cv2.imshow("hsv_edges", hsv_edges)
    return hsv_img, bgr_img

time0 = time.time()
# Run analyzeImage on a test image
isTesting = True
if(not isTesting):
    hsv_img, bgr_img = analyzeImage("/Users/cbmonk/AnacondaProjects/Advanced-Self-Driving-Car/TestImages/52.png")
    cv2.imshow("HSV", hsv_img)
    cv2.imshow("BGR", bgr_img)
    cv2.waitKey(0)
else:
    for img in range(1,55):
        path = "/Users/cbmonk/AnacondaProjects/Advanced-Self-Driving-Car/TestImages/"+str(img)+".png"
        hsv_img, bgr_img = analyzeImage(path)
        cv2.imshow("HSV", hsv_img)
        cv2.imshow("BGR", bgr_img)
        cv2.waitKey(0)
print("Total time:", time.time()-time0)

# Close everything out when any key is pressed
cv2.destroyAllWindows()
