# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 21:41:00 2020

@author: ganya
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

flimit = 220
slimit = 255

a4_shape = (877, 620)

def fupdate(value):
    global flimit
    flimit = value


def supdate(value):
    global slimit
    slimit = value


def get_dist(x2, x1, y2, y1):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)


def get_sheet_shape(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]

    width1 = get_dist(p3[0], p4[0], p3[1], p4[1])
    width2 = get_dist(p2[0], p1[0], p2[1], p1[1])

    height1 = get_dist(p2[0], p3[0], p2[1], p3[1])
    height2 = get_dist(p1[0], p4[0], p1[1], p4[1])

    w = max(int(width1), int(width2))
    h = max(int(height1), int(height2))

    if w < h:
        ratio = w/a4_shape[0]
        h = int(ratio * a4_shape[1])
        return h, w
    else:
        ratio = h/a4_shape[0]
        w = int(ratio * a4_shape[1])
        return h, w


def order_points(pts):
    result = np.zeros((4, 2), dtype="f4")

    s = pts.sum(axis=1)
    result[0] = pts[np.argmin(s)]  # top-left
    result[2] = pts[np.argmax(s)]  # bottom-right

    s = np.diff(pts, axis=1)
    result[1] = pts[np.argmin(s)]  # top-right
    result[3] = pts[np.argmax(s)]  # bottom-left

    return result


cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.namedWindow("Mask", cv2.WINDOW_KEEPRATIO)

cv2.createTrackbar('f', 'Mask', flimit, 255, fupdate)
cv2.createTrackbar('s', 'Mask', slimit, 255, supdate)

template = cv2.imread('picture.jpg')
gray_pic = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

kernel = np.ones((7, 7))

while cam.isOpened():
    ret, frame = cam.read()
    
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(converted, np.array([80, flimit, 0]), np.array([200, slimit, 255]))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)
    
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    cnts, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if len(cnts) > 0:

            paper = max(cnts, key=cv2.contourArea)
            
            eps = 0.1 * cv2.arcLength(paper, True)
            approx = cv2.approxPolyDP(paper, eps, True)
            cv2.drawContours(frame, [approx], -1, (102, 126, 234), 2)
            for p in approx:
                cv2.circle(frame, tuple(*p), 2, (0, 255, 0), 1)

            if len(approx) == 4:
                cv2.drawContours(frame, [approx], -1, (118, 75, 162), 3)
                pts = approx.reshape(4, 2)
                pts = order_points(pts)

                cols, rows = get_sheet_shape(pts)

                pts2 = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype="float32")

                M = cv2.getPerspectiveTransform(pts, pts2)
                warp = cv2.warpPerspective(frame, M, (cols, rows))
                
                gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
                res = cv2.matchTemplate(gray, gray_pic, cv2.TM_CCORR)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                top_left = max_loc
                bottom_right = (top_left[0] + gray_pic.shape[1], top_left[1] + gray_pic.shape[0])
                
                threshold = 850000000
                if np.amax(res) > threshold:
                    cv2.putText(frame, "found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0))
                else:
                    cv2.putText(frame, "not found", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0))
                
                cv2.rectangle(warp, top_left, bottom_right, (0, 0, 255), 5)
                cv2.imshow("Sheet", warp)

    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.imwrite("screenshot.png", frame)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()