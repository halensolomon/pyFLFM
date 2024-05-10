import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def view_grabber(img, num_circles, fill_ratio):
    ''' This function breaks up the image into distinct regions for the user to mix and match later on.
    Args:
        img (numpy.ndarray): Image to be divided into regions.
        num_circles_row (array of ints): Number of circles seen in image. (Helps with the division of the image)
        fill_ratio (float): Ratio of the circle's radius to the image's width.
    '''
    
    img_long = max(img.shape)
    num_circle_small = max(num_circle.shape)
    
    # Calculate the expected radius of the circles
    expected_radius = int(img.shape[1]*fill_ratio/(2*num_circles_small))
    
    # Hough Transform
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0.8*expected_radius, maxRadius=1.2*expected_radius)
    s
    # Draw circles
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0,:]:
            cv.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
            
    # Show user the image with circles
    cv.imshow('Circles', img)
    cv.waitKey(0)
    
    # Ask user if they want to continue
    print('Found', len(circles[0,:]), 'circles')
    print('Would you like to continue?')
    print('Press "q" to quit, or "c" to continue')
    if cv.waitKey(0) == ord('q'):
        cv.destroyAllWindows()
        sys.exit("User quit the program.")

    else:
        # Ask user if they would like to remove any circles
        print('Would you like to remove any circles?')
        print('Press "r" to remove a circle, or "c" to continue')
        if cv.waitKey(0) == ord('r'):
            print('Click on the circle you would like to remove')
            cv.destroyAllWindows()
            cv.imshow('Circles', img)
            cv.setMouseCallback('Circles', remove_circle)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            cv.destroyAllWindows()
        
        cv.destroyAllWindows()
        return circles
    
    return None
    
    
    
