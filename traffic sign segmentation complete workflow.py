import os
import sys
assert sys.version_info >= (3,7)
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import cv2 as cv
import matplotlib.pyplot as plt
import time
import imutils

# For reproducibility,
np.random.seed(99)

# Make sure that optimization is enabled
if not cv.useOptimized():
    cv.setUseOptimized(True)
cv.useOptimized()

# Traffic signs with blue colour
blue_ts = (20,21,22,23,24,25,26,27,28,29,30,31)

# Traffic signs with red colour
red_ts = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,33,52,53,54,55,56,57)

# Traffic signs with yellow colour
yellow_ts = (18,19,32,34,335,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51)

# All traffic signs
all_ts = tuple(range(0, 58))

# Bounderies for HSV thresholding, fine tuned to fit the dataset
# Red colour segmentation
lower_red, upper_red = (0, 25, 10), (10, 255, 255)
lower_red2, upper_red2 = (150, 25, 0), (180, 255, 255)

# Blue colour segmentation
lower_blue, upper_blue = (90, 60, 10), (135, 255, 255)

# Yellow colour segmentation
lower_yellow, upper_yellow = (10, 50, 10), (55, 255, 255)
lower_black, upper_black = (0, 0, 0), (180, 50, 80)

def hsv_edges_bg_mask(image):
    image = cv.GaussianBlur(image, (5,5), 3)
    h, w = image.shape[:2]
    noise_mask = np.ones((h+2, w+2))
    
    
    # Get the position to fill
    left = int(w*0.1)
    right = int(w*0.9)
    top = int(h*0.1)
    bottom = int(h*0.9)
    
    
    # Get colour of edges
    top_left_colour = image[top,left]
    bottom_left_colour = image[bottom, left]
    top_right_colour = image[top,right]
    bottom_right_colour = image[bottom,right]
    
    
    # Get all the noise of the background
    mask = cv.inRange(image, top_left_colour - 7, top_left_colour + 7)
    mask += cv.inRange(image, bottom_left_colour - 7, bottom_left_colour + 7)
    mask += cv.inRange(image, top_right_colour - 7, top_right_colour + 7)
    mask += cv.inRange(image, bottom_right_colour - 7, bottom_right_colour + 7)
    
    
    return mask

def canny_seg(image, dilate_kernel1, dilate_kernel2, dilate_iterations):
    # Smoothing the image
    image = cv.GaussianBlur(image, (7, 7), 1)
    
    
    # Canny segmentation
    mask = cv.Canny(image, 1, 100)
    
    
    # Dilate the edges found by canny segmentation
    mask = cv.dilate(mask, (dilate_kernel1,dilate_kernel2), iterations=dilate_iterations)
    
    
    return mask

def blue_seg(image_hsv, image_bgr, background_mask):
    # Thresholding
    mask = cv.inRange(image_hsv, lower_blue, upper_blue)
    
    # Removing edges
    canny_mask = canny_seg(image_bgr, 3, 3, 2)
    
    
    final_mask = mask - canny_mask - background_mask
    
    
    return final_mask, mask, canny_mask

def red_seg(image_hsv, image_bgr, background_mask):
    # Thresholding
    mask = cv.inRange(image_hsv, lower_red, upper_red)
    mask += cv.inRange(image_hsv, lower_red2, upper_red2)
    
    
    # Removing edges
    canny_mask = canny_seg(image_bgr, 1, 1, 1)
    
    
    final_mask = mask - canny_mask - background_mask
    
    
    return final_mask, mask, canny_mask

def black_seg(image_hsv, image_bgr, background_mask):
    h,w = image_hsv.shape[:2]
    
    
    # Segment the edges
    canny_mask = canny_seg(image_bgr, 1, 1, 1)

    
    # edged is the edge detected image
    cnts = cv.findContours(canny_mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv.contourArea, reverse = True)[:5]
    # loop over the contours
    i = 0
    for c in cnts:
        if (cv.contourArea(c) < h * w * 0.10):
            break

        # approximate the contour
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.1 * peri, True)
        # Contour has 3 points, represents a triangle
        if len(approx) == 3:
            temp_mask = np.zeros(image_hsv.shape[:2], dtype = np.uint8)
            cv.drawContours(temp_mask, cnts, i, (255,255,255), -1)
            return temp_mask - background_mask, temp_mask, canny_mask
        i+=1
    
    
    mask = cv.inRange(image_hsv, lower_black, upper_black)
    mask += cv.inRange(image_hsv, lower_yellow, upper_yellow)
    
    
    final_mask = mask - canny_mask - background_mask
    
    
    return final_mask, mask, canny_mask

def find_largest_contour(image, fill = -1):
    # Threshold to keep only the bright area in the mask 
    _, image = cv.threshold(image, 200, 255, cv.THRESH_TOZERO)
    
    mask = np.zeros(image.shape, dtype = np.uint8)
    largest_contour = None
    
    
    # Find all contours
    if (int(cv.__version__[0]) > 3):
        contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    else:
        _, contours, hierarchy = cv.findContours(image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    
    
    # Find and draw largest contour
    if len(contours) != 0:
        cnt_list = np.zeros(len(contours))
        for i in range(0,len(contours)):
            cnt_list[i] = cv.contourArea(contours[i])

            
        largest_contour_index = np.argmax(cnt_list)
        largest_contour = contours[largest_contour_index]
        cv.drawContours(mask, contours, largest_contour_index, (255,255,255), fill)

        
    return largest_contour, mask

def draw_bounding_box(image, largest_contour, colour):
    bounding_box = image.copy()
    
    
    # If contour is not exists
    if (largest_contour is None):
        return bounding_box, (0,0,0,0)
    
    
    cv.boundingRect(largest_contour)
    x,y,w,h = cv.boundingRect(largest_contour)
    cv.rectangle(bounding_box,(x,y),(x+w,y+h),colour,2)
    return bounding_box, (x,y,x+w,y+h)

def closest_contour(contours, image):
    h, w = image.shape[:2]
    center = np.array((int(h/2), int(w/2)))
    rect = np.zeros((h, w), dtype = np.uint8)
    
    
    closest = 0
    closest_distance = 9999
    
    
    if(contours[0] is not None):
        blue_dist = cv.moments(contours[0])
        cx = int(blue_dist['m10']/max(blue_dist['m00'], 1))
        cy = int(blue_dist['m01']/max(blue_dist['m00'], 1))
        blue_distance = ((cx - center[1])**2 + (cy - center[0])**2)**0.5
        if blue_distance < closest_distance:
            _,_,w0,y0 = cv.boundingRect(contours[0])
            if(w0*y0 > h*w*0.2):
                closest_distance = blue_distance
    
    
    if(contours[1] is not None):
        red_dist = cv.moments(contours[1])
        cx = int(red_dist['m10']/max(red_dist['m00'], 1))
        cy = int(red_dist['m01']/max(red_dist['m00'], 1))
        red_distance = ((cx - center[1])**2 + (cy - center[0])**2)**0.5
        if red_distance < closest_distance:
            _,_,w1,y1 = cv.boundingRect(contours[1])
            if(w1*y1 > h*w*0.2):
                closest = 1
                closest_distance = red_distance
        
        
    if(contours[2] is not None):
        yellow_dist = cv.moments(contours[2])
        cx = int(yellow_dist['m10']/max(yellow_dist['m00'], 1))
        cy = int(yellow_dist['m01']/max(yellow_dist['m00'], 1))
        yellow_distance = ((cx - center[1])**2 + (cy - center[0])**2)**0.5
        if yellow_distance < closest_distance:
            _,_,w2,y2 = cv.boundingRect(contours[2])
            if(w2*y2 > h*w*0.2):
                closest = 2
        
    
    return closest

def segmentation(annotation, images, traffic_sign, save_result):
    seconds = time.time()
    ts = np.where(annotation['Category'].isin(traffic_sign))[0]
    combine_all = np.empty((150,1200,3))
    bounding_box_points = []
    accuracies = [0,0,0,0,0]
    
    for i in range (len(ts)):
        # Create a copy of the original image
        image = images[ts[i]].copy()
        h, w = image.shape[:2]
        
        
        # Convert image into hsv colour space and gray colour space
        image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        
        # Segment background colour
        background_mask = hsv_edges_bg_mask(image_hsv)
#         background_mask = np.zeros(image.shape[:2], dtype = np.uint8)
 
        # HSV segmentation
        blueseg_mask, blue_mask, blue_canny_mask = blue_seg(image_hsv, image, background_mask)
        redseg_mask, red_mask, red_canny_mask = red_seg(image_hsv, image, background_mask)
        yellowseg_mask, yellow_mask, yellow_canny_mask = black_seg(image_hsv, image, background_mask)
        
        
        # Find largest contour
        blue_largest_contour, blue_largest_contour_mask = find_largest_contour(blueseg_mask)
        red_largest_contour, red_largest_contour_mask = find_largest_contour(redseg_mask)
        yellow_largest_contour, yellow_largest_contour_mask = find_largest_contour(yellowseg_mask)
        
        
        # Draw bounding box
        blue_bounding_box, blue_box_points = draw_bounding_box(image, blue_largest_contour, (0,255,0))
        red_bounding_box, red_box_points = draw_bounding_box(image, red_largest_contour, (0,255,0))
        yellow_bounding_box, yellow_box_points = draw_bounding_box(image, yellow_largest_contour, (0,255,0))
        
        
        # Find the contour the is closest to the center
        contours = [blue_largest_contour, red_largest_contour, yellow_largest_contour]
        closest = closest_contour(contours, image)
        
    
        if(closest == 0):
            canny_mask = blue_canny_mask
            hsv_mask = blue_mask
            segmented_mask = blueseg_mask
            final_mask = blue_largest_contour_mask
            bounding_box = blue_bounding_box
            bounding_box_point = blue_box_points
        elif (closest == 1):
            canny_mask = red_canny_mask
            hsv_mask = red_mask
            segmented_mask = redseg_mask
            final_mask = red_largest_contour_mask
            bounding_box = red_bounding_box
            bounding_box_point = red_box_points
        else:
            canny_mask = yellow_canny_mask
            hsv_mask = yellow_mask
            segmented_mask = yellowseg_mask
            final_mask = yellow_largest_contour_mask
            bounding_box = yellow_bounding_box
            bounding_box_point = yellow_box_points


        bounding_box_points.append(bounding_box_point)
        
        
        # calculating and storing the accuracy
        area1 = (bounding_box_point[2] - bounding_box_point[0]) * (bounding_box_point[3] - bounding_box_point[1])
        area2 = (train_annotation['End_X'][i] - train_annotation['Start_X'][i]) * (train_annotation['End_Y'][i] - train_annotation['Start_Y'][i])
        X_diff = max(0, min(bounding_box_point[2], train_annotation['End_X'][i]) - max(bounding_box_point[0], train_annotation['Start_X'][i]))
        Y_diff = max(0, min(bounding_box_point[3], train_annotation['End_Y'][i]) - max(bounding_box_point[1], train_annotation['Start_Y'][i]))
        overlapp = X_diff * Y_diff
        area = area1 + area2 - overlapp
        accuracy = round(overlapp/area*100,2)
        if accuracy >= 95:
            accuracies[0] += 1
        elif accuracy >= 90:
            accuracies[1] += 1
        elif accuracy >= 85:
            accuracies[2] += 1
        elif accuracy >= 80:
            accuracies[3] += 1
        else:
            accuracies[4] += 1
        
        
        if(save_result):
            canny_mask = np.stack((canny_mask,)*3, axis=-1)
            background_mask = np.stack((background_mask,)*3, axis=-1)
            hsv_mask = np.stack((hsv_mask,)*3, axis=-1)
            segmented_mask = np.stack((segmented_mask,)*3, axis=-1)
            final_mask = np.stack((final_mask,)*3, axis=-1)
            combine = np.concatenate((image, canny_mask, background_mask, hsv_mask, segmented_mask, final_mask, bounding_box), axis = 1)
            combine = cv.resize(combine, (1050,150), interpolation = cv.INTER_AREA)
            combine = np.concatenate((combine, np.zeros((150, 150, 3))), axis = 1)
            
            combine = np.concatenate((np.zeros((30, 1200, 3)), combine), axis = 0)
            cv.putText(combine, ("original image"), (5, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("edges detected"), (155, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("background"), (305, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("HSV mask"), (455, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("segmentation"), (605, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("largest contour"), (755, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("bounding box"), (905, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, ("accuracy"), (1055, 15), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            cv.putText(combine, str(accuracy) + "%", (1100, 110), 0, 0.5, (255,255,255), 1, cv.LINE_AA)
            
#             if (accuracy > 90):
#                 cv.imwrite(os.getcwd() + "\\put in report\\" + 'saved' + str(i) + '.jpg', combine)
            
            combine_all = np.concatenate((combine_all, combine), axis = 0)      
            if (i%100 == 0):
                cv.imwrite('saved' + str(i//100) + '.jpg', combine_all)
                combine_all = combine
                print(i, "/", len(ts), "Elapsed time:", time.time() - seconds, "Average time:", (time.time() - seconds)/(i+1))
        elif (i%100 == 0):
            print(i, "/", len(ts), "Elapsed time:", time.time() - seconds, "Average time:", (time.time() - seconds)/(i+1))
    
    return bounding_box_points, accuracies

# Path to the train set annotation txt file
path = os.path.join(os.getcwd(), 'TSRD-Train Annotation\\TsignRecgTrain4170Annotation.txt')

# Name of the columns
columns = ['File Name', 'Width', 'Height', 'Start_X', 'Start_Y', 'End_X', 'End_Y', 'Category']

# Read the content of the train annotation txt file into train_annotation
train_annotation = pd.read_csv(path, sep=";", index_col = False, names = columns)

# One hot encode the category of the images
encoder = LabelBinarizer()
train_category = encoder.fit_transform(train_annotation['Category'])

#path to the folder containing the trains set image files
path = os.path.join(os.getcwd(), 'TSRD-train')

#read the train images into train_image
train_images = []
for i in range (train_annotation.shape[0]):
    train_images.append(cv.imread(path + '\\' + train_annotation.iloc[i][0]))

bounding_box_points, accuracies = segmentation(train_annotation, train_images, all_ts, save_result = False)

total_overlapp = 0
area = 0
for i in range (len(bounding_box_points)):
    area1 = (bounding_box_points[i][2] - bounding_box_points[i][0]) * (bounding_box_points[i][3] - bounding_box_points[i][1])
    area2 = (train_annotation['End_X'][i] - train_annotation['Start_X'][i]) * (train_annotation['End_Y'][i] - train_annotation['Start_Y'][i])
    X_diff = max(0, min(bounding_box_points[i][2], train_annotation['End_X'][i]) - max(bounding_box_points[i][0], train_annotation['Start_X'][i]))
    Y_diff = max(0, min(bounding_box_points[i][3], train_annotation['End_Y'][i]) - max(bounding_box_points[i][1], train_annotation['Start_Y'][i]))
    overlapp = X_diff * Y_diff
    total_overlapp += overlapp
    area = area + area1 + area2 - overlapp
print("Segmentation accuracy:", total_overlapp/area)
print("Total image:", str(accuracies[0] + accuracies[1] + accuracies[2] + accuracies[3] + accuracies[4]))
print("Accuracy between 95 - 100:", str(accuracies[0]))
print("Accuracy between 90 - 95:", str(accuracies[1]))
print("Accuracy between 85 - 90:", str(accuracies[2]))
print("Accuracy between 80 - 85:", str(accuracies[3]))
print("Accuracy lower than 80:", str(accuracies[4]))
