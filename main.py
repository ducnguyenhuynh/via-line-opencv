import cv2
import sys
import json
import time
import socket
import threading

from utils import *
from parameters import Parameter
from image_processing import *

p = Parameter()

def test_video(url):
    cap = cv2.VideoCapture(url)
    while cap.isOpened():
        ret, image_ori = cap.read()
        point_in_lane = get_point_in_lane(image_ori)
        
        ##
        binary_image = get_binary_pipeline(image_ori)
        bird_view_image = warp_image(binary_image)  
        # cv2.circle(bird_view_image, (p.point_in_lane[0], p.point_in_lane[1]), 7, 1,8)
        lfit, rfit = track_lanes_initialize(bird_view_image)
        lfit, rfit =  adjust_fits(lfit, rfit, point_in_lane)
        cfit = get_center_fit(lfit, rfit)

        colored_lane, center_line = lane_fill_poly(bird_view_image, image_ori, cfit, lfit, rfit)
        ##
        cv2.imshow("image", image_ori)
        cv2.imshow("binary_image", bird_view_image*255) 
        cv2.imshow("lane", colored_lane) 

        key = cv2.waitKey(1)
        binary_image 
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_video("map_1.avi")