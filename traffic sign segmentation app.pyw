import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import tkinter.messagebox as msgbox
from tkinter import font
from PIL import ImageTk, Image
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

# Extract background colours
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

# detect edges
def canny_seg(image, dilate_kernel1, dilate_kernel2, dilate_iterations):    
    # Smoothing the image
    image = cv.GaussianBlur(image, (7, 7), 1)
    
    # Canny segmentation
    mask = cv.Canny(image, 1, 100)
    
    # Dilate the edges found by canny segmentation
    mask = cv.dilate(mask, (dilate_kernel1,dilate_kernel2), iterations=dilate_iterations)
    
    return mask

# extract blue pixels
def blue_seg(image_hsv, image_bgr, background_mask):
    # Thresholding
    mask = cv.inRange(image_hsv, lower_blue, upper_blue)
    
    # Removing edges
    canny_mask = canny_seg(image_bgr, 3, 3, 2)
    
    final_mask = mask - canny_mask - background_mask
    
    return final_mask, mask, canny_mask

# extract red pixels
def red_seg(image_hsv, image_bgr, background_mask):
    # Thresholding
    mask = cv.inRange(image_hsv, lower_red, upper_red)
    mask += cv.inRange(image_hsv, lower_red2, upper_red2)
    
    
    # Removing edges
    canny_mask = canny_seg(image_bgr, 1, 1, 1)
    
    
    final_mask = mask - canny_mask - background_mask
    
    
    return final_mask, mask, canny_mask

# detect triangles + extract yellow and black pixels
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

# find largest contour
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

# draw bounding box
def draw_bounding_box(image, largest_contour, colour):
    bounding_box = image.copy()
    
    # If contour is not exists
    if (largest_contour is None):
        return bounding_box, (0,0,0,0)
    
    cv.boundingRect(largest_contour)
    x,y,w,h = cv.boundingRect(largest_contour)
    cv.rectangle(bounding_box,(x,y),(x+w,y+h),colour,2)
    return bounding_box, (x,y,x+w,y+h)

# find contour closest to center
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
            # need to have area of at least 20% area of image
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

def segmentation(image):
    seconds = time.time()
    h, w = image.shape[:2]
    
    # Convert image into hsv colour space and gray colour space
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        
    # Segment background colour
    background_mask = hsv_edges_bg_mask(image_hsv)
    
    # HSV segmentation
    blueseg_mask, _, _ = blue_seg(image_hsv, image, background_mask)
    redseg_mask, _, _ = red_seg(image_hsv, image, background_mask)
    yellowseg_mask, _, _ = black_seg(image_hsv, image, background_mask)
    
    # Find largest contour
    blue_largest_contour, _ = find_largest_contour(blueseg_mask)
    red_largest_contour, _ = find_largest_contour(redseg_mask)
    yellow_largest_contour, _ = find_largest_contour(yellowseg_mask)
    
    # Draw bounding box
    blue_bounding_box, _ = draw_bounding_box(image, blue_largest_contour, (255,0,0))
    red_bounding_box, _ = draw_bounding_box(image, red_largest_contour, (0,0,255))
    yellow_bounding_box, _ = draw_bounding_box(image, yellow_largest_contour, (0,255,255))
    
    # Find the contour the is closest to the center
    contours = [blue_largest_contour, red_largest_contour, yellow_largest_contour]
    closest = closest_contour(contours, image)
    
    if(closest == 0):
        return blue_bounding_box
    elif (closest == 1):
        return red_bounding_box
    else:
        return yellow_bounding_box


class Application(tk.Tk):
    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Traffic Sign Sementation App')
        
        # Create a container
        container = tk.Frame(self) 
        container.pack(side = "top", fill = "both", expand = True)
        container.grid_rowconfigure(0, weight = 1)
        container.grid_columnconfigure(0, weight = 1)
  
        # Store frames to an empty array
        self.frames = {} 
        for F in (SegmentationPage,):
            frame = F(container, self)
            self.frames[F] = frame
            frame.grid(row = 0, column = 0, sticky ="nsew")
        self.show_frame(SegmentationPage)
  
    # Switch frame
    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()

class SegmentationPage(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.maxhw = 500
        self.filename = None
        
        self.ori_image = ttk.Label(self, width = 20)
        self.ori_image.grid(row = 0, column = 0, padx = 10, pady = 10)
        
        self.segmented_image = ttk.Label(self, width = 20)
        self.segmented_image.grid(row = 0, column = 1, padx = 10, pady = 10)
        
        self.notice = ttk.Label(self, text ="-", font = ("Verdana", 15))
        self.notice.grid(row = 1, column = 0, columnspan = 2, padx = 10, pady = 10)
        
        style = ttk.Style()
        style.configure('my.TButton', font=("Verdana", 15))
        AddPersonBtn = ttk.Button(self, text ="Choose image", style='my.TButton', command = self.ChooseFile)
        AddPersonBtn.grid(row = 2, column = 0, padx = 10, pady = 10)
        AddPersonBtn = ttk.Button(self, text ="Segmentate", style='my.TButton', command = self.Segmentate)
        AddPersonBtn.grid(row = 2, column = 1, padx = 10, pady = 10)

    def ChooseFile(self):
        filename = askopenfilename()
        self.filename = filename
        image = cv.imread(filename)
        
        width = image.shape[1]
        if(width > self.maxhw):
            height = image.shape[0]
            scale = width/self.maxhw
            image = cv.resize(image, (self.maxhw, int(height/scale)))
        
        height = image.shape[0]
        if(height > self.maxhw):
            width = image.shape[1]
            scale = height/self.maxhw
            image = cv.resize(image, (int(width/scale), self.maxhw))
            
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(image=img)
        self.ori_image.imgtk = img
        self.ori_image.configure(image = img)
        
    def Segmentate(self):
        image = segmentation(cv.imread(self.filename))
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        width = image.shape[1]
        if(width > self.maxhw):
            height = image.shape[0]
            scale = width/self.maxhw
            image = cv.resize(image, (self.maxhw, int(height/scale)))
        height = image.shape[0]
        if(height > self.maxhw):
            width = image.shape[1]
            scale = height/self.maxhw
            image = cv.resize(image, (int(width/scale), self.maxhw))
        img = Image.fromarray(image)
        img = ImageTk.PhotoImage(image=img)
        self.segmented_image.imgtk = img
        self.segmented_image.configure(image = img)
        self.notice.config(text = "Success")
        
app = Application()
app.mainloop()