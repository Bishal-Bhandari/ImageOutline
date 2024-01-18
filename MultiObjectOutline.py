import json
import cv2
import numpy as np

img = cv2.imread('Image/img_1.png')  # read image
bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert img to BnW
gus_blr = cv2.GaussianBlur(bnw, (5, 5), 0)  # decreasing the noise
edges = cv2.Canny(gus_blr, 50, 150)  # edge detection

contour_img, hire_rel = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # finding the contour

if contour_img:  # validation for contour
    all_contour_points = []
    for contour in contour_img:
        contour_points = contour.reshape(-1, 2)  # representing into 2D array
        all_contour_points.append(contour_points.tolist())  # rows to list added
    with open('contoursFile.json', 'w') as f:
        json.dump(all_contour_points, f)  # save json

    cv2.drawContours(img, contour_img, -1, (0, 255, 0), 4)  # drawing the outline

    cv2.imshow('Image with Contours', img)  # display
    cv2.waitKey(0)  # wait until key press
    cv2.destroyAllWindows()
else:
    print("No contours found.")
