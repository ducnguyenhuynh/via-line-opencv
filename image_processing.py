import cv2
import numpy as np 
from parameters import Parameter
p = Parameter()


def add_filter_RGB(image, threshR=(0,255), threshG=(0,255), threshB=(0,255), show = False):
    R = image[:,:,2]
    G = image[:,:,1]
    B = image[:,:,0]
    binary_output = np.zeros_like(R)
    binary_output[(R >= threshR[0]) & (R <= threshR[1]) & (G >= threshG[0]) & (G <= threshG[1]) & (B >= threshB[0]) & (B <= threshB[1])] = 1
    if show:
        return binary_output*255
    return binary_output

def get_binary_pipeline(image, show = False):
    image_cp =  cv2.GaussianBlur(image,(3,3), 0)
    binary = add_filter_RGB(image_cp, (200,255),(200,255),(200,255))
    binary_in_shadow = add_filter_RGB(image_cp, (50, 90), (60,120), (120,150))
    output = cv2.bitwise_or(binary, binary_in_shadow)
    if show:
        return output*255
    return output

def add_filter_hsv(img, lower=np.array([10, 0, 0]), upper =np.array([180, 50,210])):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, lower, upper)
    return mask

def warp_image(image):
    warped_image = cv2.warpPerspective(image, p.perspective_transform, p.image_size, flags=cv2.INTER_LINEAR)
    return warped_image



