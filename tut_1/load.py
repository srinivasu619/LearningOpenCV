import cv2
import numpy as np
from matplotlib import pyplot as plot
img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
cv2.imshow('IMAGE',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
