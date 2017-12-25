import numpy as np
import cv2

IMG = cv2.imread('../tut_1/watch.jpg', cv2.IMREAD_COLOR)

# accessing the pixel of the image
print(IMG[55, 55])

# accessing a block of pixels
IMG[100:105, 100:105] = [255, 255, 255]

# copying a block of an image
BLOCK = IMG[80:130, 50:100]
IMG[0:50, 0:50] = BLOCK
cv2.imshow('IMAGE',IMG)
cv2.waitKey(0)
cv2.destroyAllWindows()