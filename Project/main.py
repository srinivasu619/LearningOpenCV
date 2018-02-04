import numpy as np
import cv2
import math

# Threshold for binary threshold
threshold = 60
# Blurring constant
blurValue = 41
# Background Subtraction Model(KNN, MOG2)
bgModel = cv2.createBackgroundSubtractorKNN(500, 100, True)
# Assigning Camera
camera = cv2.VideoCapture(0)
# Set Frame rate as 10
camera.set(cv2.CAP_PROP_FPS, 10)

while camera.isOpened():
   ret, frame = camera.read()
   # Flip frame horizontally
   xFrame = cv2.flip(frame, 1)
   # convert to grayScale
   grayScaleFrame = cv2.cvtColor(xFrame, cv2.COLOR_RGB2GRAY)
   # apply background subtraction
   backgroundSubtractedFrame = bgModel.apply(xFrame)
   backgroundMaskedFrame = cv2.bitwise_and(
       xFrame, xFrame, mask=backgroundSubtractedFrame)
   grayFrame = cv2.cvtColor(backgroundMaskedFrame, cv2.COLOR_BGR2GRAY)
   # perform Blur, Dilation and Erosion
   blurFrame = cv2.GaussianBlur(grayFrame, (blurValue, blurValue), 0)
   kernel = np.ones((5, 5), np.uint8)
   blurFrame = cv2.erode(blurFrame, kernel, iterations=1)
   blurFrame = cv2.dilate(blurFrame, kernel, iterations=1)
   # Binary threshold
   ret2, thresh = cv2.threshold(blurFrame, threshold, 255, cv2.THRESH_BINARY)
   cv2.imshow("Threshold", thresh)
   _, contours, hierarchy = cv2.findContours(
       thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   length = len(contours)
   maxArea = -1
   if length > 0:
       try:
           for i in range(length):
               temp = contours[i]
               #cv2.imshow("DrawThres", thresh)
               area = cv2.contourArea(temp)
               if area > maxArea:
                   maxArea = area
                   ci = i
           # largest contour
           cnt = contours[ci]
           # mask contour area on black canvas
           # Create mask where white is what we want, black otherwise
           mask = np.zeros_like(xFrame)
           # Draw filled contour in mask
           cv2.drawContours(mask, contours, ci, (255, 255, 255), -1)
           # Extract out the object and place into output image
           out = np.zeros_like(xFrame)
           out[mask == 255] = xFrame[mask == 255]
           #cv2.imshow("Output", out)
           # convex hull
           hull = cv2.convexHull(cnt)
           d_hull = cv2.convexHull(cnt,returnPoints = False)
           defects = cv2.convexityDefects(cnt, d_hull)
           cv2.drawContours(xFrame, [cnt], 0, (0, 255, 0), 2)
           cv2.drawContours(xFrame, [hull], 0, (0, 0, 255), 3)
           M = cv2.moments(cnt)
           cx = int(M['m10']/M['m00'])
           cy = int(M['m01']/M['m00'])
           centroid = tuple([cx,cy])
           cv2.circle(xFrame, centroid, 5, [170,232,238], -1)
           for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]

                start = tuple(cnt[s][0])
                end = tuple(cnt[e][0])
                far = tuple(cnt[f][0])
                a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
                if angle <= 80:
                    cv2.circle(xFrame, far, 5, [255,0,0], -1)
                    cv2.circle(xFrame, start, 5, [147,20,255], -1)
                    cv2.line(xFrame,start, far, [255,255,255], 2)

           cv2.imshow("DRAWN", xFrame)
       except IndexError:
           print("NO Counter Found")
   k = cv2.waitKey(30) & 0xff
   if k == 27:
       break
# release camera resource
camera.release()
cv2.destroyAllWindows()
