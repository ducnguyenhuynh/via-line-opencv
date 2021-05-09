import cv2
import numpy as np 
class Parameter():

    image_size = (640, 480)
    x = image_size[0]
    y = image_size[1]
    source_points = np.float32([
    [0, y],
    [0, (7/9)*y+10],
    [x, (7/9)*y+10],
    [x, y]
    ])
    
    destination_points = np.float32([
    [0.25 * x, y],
    [0.25 * x, 0],
    [x - (0.25 * x), 0],
    [x - (0.25 * x), y]
    ])

    perspective_transform = cv2.getPerspectiveTransform(source_points, destination_points)
    inverse_perspective_transform = cv2.getPerspectiveTransform( destination_points, source_points)
    
    # point_in_lane = [0,0]