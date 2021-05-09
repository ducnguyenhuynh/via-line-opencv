import cv2
import numpy as np 
from image_processing import *

def get_point_in_lane(image):
    warped = warp_image(image)
    lane_image = add_filter_hsv(warped)
    lane_shadow = add_filter_RGB(warped, (45,55), (55,70), (60,80))
    lane = cv2.bitwise_or(lane_image, lane_shadow)
    
    histogram_x = np.sum(lane[:,:], axis=0)
    histogram_y = np.sum(lane[:,:], axis=1)
    lane_x = np.argmax(histogram_x)
    lane_y = np.argmax(histogram_y)
    for y in range(lane_y,0,-1):
        if lane[y][lane_x] == 255:
            # cv2.circle(image, (y, lane_x), 50, (255,0,0), 3)
            return [lane_y, lane_x]
    # return [0, 0]


def get_val(y,poly_coeff):
    return poly_coeff[0]*y**2+poly_coeff[1]*y+poly_coeff[2]

def track_lanes_initialize(binary_warped):   

    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint+100])
    rightx_base = np.argmax(histogram[midpoint+100:]) + midpoint+100
    nwindows = 9
    window_height = np.int(binary_warped.shape[0]/nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 60
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []  
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = int(binary_warped.shape[0] - (window+1)*window_height)
        win_y_high = int(binary_warped.shape[0] - window*window_height)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 3) 
        # cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 3) 
        # cv2.imshow('out_img',out_img)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        
    
    left_lane_inds,right_lane_inds = check_lane_inds(left_lane_inds,right_lane_inds)
    if len(left_lane_inds) != 0:
        left_lane_inds = np.concatenate(left_lane_inds)
    if len(right_lane_inds) !=0:
        right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    left_fit = np.array([])
    right_fit = np.array([])
    if len(leftx) != 0:
        left_fit  = np.polyfit(lefty, leftx, 2)
    if len(rightx) != 0:
        right_fit  = np.polyfit(righty, rightx, 2)
    return left_fit, right_fit


def check_lane_inds(left_lane_inds, right_lane_inds):
    countleft = 0
    countright = 0
    missing_one_line = False
    for x in range(9):
        left = np.asarray(left_lane_inds[x])
        right = np.asarray(right_lane_inds[x])
        if len(left) == 0:
            countleft+=1
        if len(right) == 0:
            countright+=1
        if len(left) == len(right) and len(left) !=0 and len(right) != 0:
            if (left == right).all():
                missing_one_line = True
    if missing_one_line:
        if countleft == countright:
            return left_lane_inds, right_lane_inds
        if countleft < countright:
            return left_lane_inds, []
        return [], right_lane_inds
    if countleft >= 6:
        return [], right_lane_inds
    if countright >= 6:
        return left_lane_inds, []
    return left_lane_inds,right_lane_inds

def check_fit_duplication(left_fit, right_fit):
    if len(left_fit) == 0 or len(right_fit) == 0:
        return left_fit, right_fit
    # print(left_fit[2], right_fit[2])
    if abs(left_fit[0] - right_fit[0]) < 0.1:
        if abs(left_fit[1] - right_fit[1]) < 0.4:
            if abs(left_fit[2] - right_fit[2]) < 30:
                return left_fit, []
    return left_fit, right_fit


def adjust_fits_ismissed(left_fit, right_fit, point_in_lane):

    # ploty = np.linspace(0, p.image_size[1] - 1, 30)
    print(point_in_lane)
    avaiable_fit = left_fit
    if len(left_fit) == 0:
        avaiable_fit = right_fit
    val = point_in_lane[1] - get_val(point_in_lane[0], avaiable_fit)
    print(val)
    if val > 0:
        print("missing right line")
        #left avaiable
        # left_fitx = get_val(ploty,avaiable_fit)
        left_fit = avaiable_fit
        right_fit = np.array([])
    else:
        print("missing left line")
        #right avaiable
        # right_fitx = get_val(ploty,avaiable_fit)
        right_fit = avaiable_fit
        left_fit = np.array([])

    return left_fit, right_fit

def adjust_fits(left_fit, right_fit, point_in_lane):
    left_fit, right_fit = check_fit_duplication(left_fit, right_fit)
    # missing 2 line
    if len(left_fit) == 0  and len(right_fit) == 0: 
        lfit_updated = np.array([])
        rfit_updated = np.array([])
        return lfit_updated, rfit_updated
    # missing 1 line
    if len(left_fit) == 0 or len(right_fit) == 0: 
        lfit_updated, rfit_updated = adjust_fits_ismissed(left_fit,right_fit, point_in_lane)
        return lfit_updated, rfit_updated
    
    return left_fit, right_fit


def get_center_fit(left_fit, right_fit):
    ploty = np.linspace(0, p.image_size[1] - 1, 30)
    # missing left
    if len(left_fit) == 0:
        rightx = get_val(ploty, right_fit)
        center_x = np.clip(rightx - 150, p.image_size[0]*0.25+1, p.image_size[0]-p.image_size[0]*0.25-1)
        center_fit = np.polyfit(ploty, center_x, 2)
        return center_fit    

    if len(right_fit) == 0:
        leftx = get_val(ploty, left_fit)
        center_x = np.clip(leftx + 150,p.image_size[0]*0.25+1,p.image_size[0]-p.image_size[0]*0.25-1)
        center_fit = np.polyfit(ploty, center_x, 2)
        return center_fit 

    leftx = get_val(ploty, left_fit)
    rightx = get_val(ploty, right_fit)
    center_x = (leftx+rightx)/2
    center_fit = np.polyfit(ploty, center_x, 2)
    return center_fit


def lane_fill_poly(binary_warped, undist, center_fit, left_fit, right_fit):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    if len(left_fit) == 0:
        left_fit = np.array([0,0,1])
    if len(right_fit) == 0:
        right_fit = np.array([0,0,binary_warped.shape[1]-1])
    left_fitx = get_val(ploty,left_fit)
    right_fitx = get_val(ploty,right_fit)
    center_fitx = get_val(ploty,center_fit)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    center_color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast x and y for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts_center = np.array([np.transpose(np.vstack([center_fitx, ploty]))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane 
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    cv2.fillPoly(center_color_warp, np.int_([pts_center]),(0,0,255))
    # Warp using inverse perspective transform
    newwarp = cv2.warpPerspective(color_warp, p.inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
    center_line = cv2.warpPerspective(center_color_warp, p.inverse_perspective_transform, (binary_warped.shape[1], binary_warped.shape[0])) 
   
    result = cv2.addWeighted(undist, 1, newwarp, 0.7, 0.3)
    result = cv2.addWeighted(result,1,center_line,0.7,0.3)
    return result, center_line
        
# if __name__ == "__main__":
#     def nothing(x):
#         pass

#     image = cv2.imread("test.png")

#     cv2.namedWindow('adjustment')
#     cv2.createTrackbar('R','adjustment',0,255,nothing)
#     cv2.createTrackbar('G','adjustment',0,255,nothing)
#     cv2.createTrackbar('B','adjustment',0,255,nothing)
#     switch = '0 : OFF \n1 : ON'
#     cv2.createTrackbar(switch, 'image',0,1,nothing)

#     while(1):
#         cv2.imshow('image', image)
#         k = cv2.waitKey(1) & 0xFF
#         if k == 27:
#             break

#         # get current positions of four trackbars
#         r = cv2.getTrackbarPos('R','adjustment')
#         g = cv2.getTrackbarPos('G','adjustment')
#         b = cv2.getTrackbarPos('B','adjustment')
#         s = cv2.getTrackbarPos(switch,'adjustment')

#         if s == 0:
#             image[:] = 0
#         else:
#             binary_output[(R >= threshR[0]) & (R <= threshR[1]) & (G >= threshG[0]) & (G <= threshG[1]) & (B >= threshB[0]) & (B <= threshB[1])] = 1
    

# cv2.destroyAllWindows()