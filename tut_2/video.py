import cv2

CAP = cv2.VideoCapture(0)

while True:
    RET, FRAME = CAP.read()
    GRAY = cv2.cvtColor(FRAME,cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',GRAY)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

CAP.release()
cv2.destroyAllWindows()