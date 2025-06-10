#### Explore Color ####


```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    print(frame.shape)

    #BGR
    R = frame[:,:,2]
    G = frame[:,:,1]
    B = frame[:,:,0]

    cv2.imshow("frame",frame)
    cv2.imshow("R",R)
    cv2.imshow("G",G)
    cv2.imshow("B",B)

    key = cv2.waitKey(1) #& 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

### Object Detection with Color Threshold ###


```python
import cv2

cap = cv2.VideoCapture(0)
while True:
    _,frame = cap.read()
    print(frame.shape)

    #BGR
    R = frame[:,:,2]
    G = frame[:,:,1]
    B = frame[:,:,0]

    cv2.imshow("frame",frame)

    obj = (R<50) & (G>50) & (B>100)
    obj = 1.0*obj
    cv2.imshow("... Detection", obj)

    key = cv2.waitKey(1) #& 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

```

### Collect Image Data 


```python
import cv2
import time

#### Read From USB Camera ####

cap = cv2.VideoCapture(0) 
while True:
    _,frame = cap.read()
    print(frame.shape)
    cv2.imshow("frame",frame)

    key = cv2.waitKey(1) #& 0xFF

    if key == ord('a'):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(filename="Images/ClassA/" + timestr + ".jpg", img = frame)

    if key == ord('b'):
        timestr = time.strftime("%Y%m%d-%H%M%S")
        cv2.imwrite(filename="Images/ClassB/" + timestr + ".jpg", img = frame)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
```


```python

```
