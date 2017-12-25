import numpy as np
import cv2

IMG = cv2.imread('../tut_1/watch.jpg', cv2.IMREAD_COLOR)

cv2.line(IMG, (0,0), (50,50), (0,255,0), 7)
PTS = np.array([[10,5], [20,30], [70,20], [50,10]], np.int32);
cv2.polylines(IMG, [PTS], True, (0,255,255), 3)
cv2.imshow('IMAGE',IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()