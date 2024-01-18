import json
import cv2
import numpy as np

img = cv2.imread('img_1.png')
bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gus_blr = cv2.GaussianBlur(bnw, (5, 5), 0)
edges = cv2.Canny(gus_blr, 50, 150)

contour_img, hire_rel = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour (assuming the outer border) based on area
largest_contour = max(contour_img, key=cv2.contourArea)
# Extract the coordinates of the outer border
border_points = largest_contour.reshape(-1, 2)
print(border_points)
list_array = border_points.tolist()
with open('contoursFile.json', 'w') as f:
    json.dump(list_array, f)

cv2.drawContours(img, contour_img, -1, (0, 255, 0), 2)

cv2.imshow('Image with Contours', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
