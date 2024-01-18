import json
import cv2
import numpy as np

img = cv2.imread('Image/img_1.png')
bnw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gus_blr = cv2.GaussianBlur(bnw, (5, 5), 0)
edges = cv2.Canny(gus_blr, 50, 150)

contour_img, hire_rel = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if contour_img:
    all_contour_points = []
    for contour in contour_img:
        contour_points = contour.reshape(-1, 2)
        all_contour_points.append(contour_points.tolist())
    with open('contoursFile.json', 'w') as f:
        json.dump(all_contour_points, f)

    cv2.drawContours(img, contour_img, -1, (0, 255, 0), 4)

    cv2.imshow('Image with Contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No contours found.")
