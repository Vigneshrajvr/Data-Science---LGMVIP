import cv2 as cv
image = cv.imread('photos/cat.jpg')
gray_image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
inverted_image=255-gray_image
blurred = cv.GaussianBlur(inverted_image,(21,21),0)
inverted_blurred = 255-blurred
pencil_sketch = cv.divide(gray_image,inverted_blurred,scale=256.0)
cv.imshow('cat_sketch', pencil_sketch)
cv.waitKey(0)