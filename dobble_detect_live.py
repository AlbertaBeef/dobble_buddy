#
# Dobble Budday - Detection/Classification (live with USB camera)
#
# References:
#   https://www.kaggle.com/grouby/dobble-card-images
#
# Dependencies:
#


import numpy as np
import cv2
import os
from datetime import datetime
import itertools

import keras
from keras.models import load_model
from keras.utils import to_categorical

from imutils.video import FPS

import dobble_utils as db

# Parameters (tweaked for video)
dir = './dobble_dataset'

scale = 1.0

global circle_minRadius
global circle_maxRadius

circle_minRadius = int(100*scale)
circle_maxRadius = int(200*scale)
circle_xxxRadius = int(250*scale)

b = int(4*scale) # border around circle for bounding box

text_fontType = cv2.FONT_HERSHEY_SIMPLEX
text_fontSize = 0.75*scale
text_color    = (0,0,255)
text_lineSize = max( 1, int(2*scale) )
text_lineType = cv2.LINE_AA

matching_x = int(10*scale)
matching_y = int(20*scale)

input_video = 0 # laptop camera
#input_video = 1 # USB webcam

displayReference = True

captureAll = False
output_dir = './output'

if not os.path.exists(output_dir):      
    os.mkdir(output_dir)            # Create the output directory if it doesn't already exist

def set_minRadius(*arg):
    global circle_minRadius
    circle_minRadius = int(arg[0])
    print("[minRadius] ",circle_minRadius)
    #pass
    
def set_maxRadius(*arg):
    global circle_maxRadius
    circle_maxRadius = int(arg[0])
    print("[maxRadius] ",circle_maxRadius)
    #pass
    
    
cv2.namedWindow('Dobble Classification')
cv2.createTrackbar('minRadius', 'Dobble Classification', circle_minRadius, circle_xxxRadius, set_minRadius)
cv2.createTrackbar('maxRadius', 'Dobble Classification', circle_maxRadius, circle_xxxRadius, set_maxRadius)


# Open video
cap = cv2.VideoCapture(input_video)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
frame_width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
frame_height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print("camera",input_video," (",frame_width,",",frame_height,")")

# Open dobble model
model = load_model('dobble_model.h5')

# Load reference images
train1_dir = dir+'/dobble_deck01_cards_57'
train1_cards = db.capture_card_filenames(train1_dir)
train1_X,train1_y = db.read_and_process_image(train1_cards,72,72)

# Load mapping/symbol databases
symbols = db.load_symbol_labels(dir+'/dobble_symbols.txt')
mapping = db.load_card_symbol_mapping(dir+'/dobble_card_symbol_mapping.txt')

print("================================")
print("Dobble Classification Demo:")
print("\tPress ESC to quit ...")
print("\tPress 'p' to pause video ...")
print("\tPress 'c' to continue ...")
print("\tPress 's' to step one frame at a time ...")
print("\tPress 'w' to take a photo ...")
print("================================")

step = False
pause = False

image = []
output = []
circle_list = []
bbox_list = []
card_list = []

frame_count = 0

# start the FPS counter
fps = FPS().start()

# init the real-time FPS counter
rt_fps_count = 0
rt_fps_time = cv2.getTickCount()
rt_fps_valid = False
rt_fps = 0.0
rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
rt_fps_x = int(10*scale)
rt_fps_y = int((frame_height-10)*scale)
    
while True:
    # init the real-time FPS counter
    if rt_fps_count == 0:
        rt_fps_time = cv2.getTickCount()

    #if cap.grab():
    if True:
        frame_count = frame_count + 1
        #flag, image = cap.retrieve()
        flag, image = cap.read()
        if not flag:
            break
        else:
            image = cv2.resize(image,(0,0), fx=scale, fy=scale) 
            output = image.copy()
            
            # detect circles in the image
            gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.medianBlur(gray1,5)
            circles = cv2.HoughCircles(gray2, cv2.HOUGH_GRADIENT, 1.5 , 100, minRadius=circle_minRadius,maxRadius=circle_maxRadius)

            circle_list = []
            bbox_list = []
            card_list = []
            
            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (cx, cy, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (cx, cy), r, (0, 255, 0), 2)
                    #cv2.rectangle(output, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)

                    # extract ROI for card
                    y1 = (cy-r-b)
                    y2 = (cy+r+b)
                    x1 = (cx-r-b)
                    x2 = (cx+r+b)
                    roi = output[ y1:y2, x1:x2, : ]
                    cv2.rectangle(output, (x1,y1), (x2,y2), (0, 0, 255), 2)
                    
                    try:
                        # dobble pre-processing
                        card_img = cv2.resize(roi,(224,224),interpolation=cv2.INTER_CUBIC)
                        card_img = card_img/255.0
                        card_x = []
                        card_x.append( card_img )
                        card_x = np.array(card_x)
                        # dobble model execution
                        card_y = model.predict(card_x)
                        # dobble post-processing
                        card_id  = np.argmax(card_y[0])
                        cv2.putText(output,str(card_id),(x1,y1-b),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
                        
                        # Add ROI to card/bbox lists
                        if card_id > 0:
                            circle_list.append((cx,cy,r))
                            bbox_list.append((x1,y1,x2,y2))
                            card_list.append(card_id)

                            if displayReference:
                                reference_img = train1_X[card_id-1]
                                reference_shape = reference_img.shape
                                reference_x = reference_shape[0]
                                reference_y = reference_shape[1]
                                output[y1:y1+reference_y,x1:x1+reference_x,:] = reference_img
                        
                    except:
                        print("ERROR : Exception occured during dobble classification ...")

                         
            if len(card_list) == 1:
                matching_text = ("[%04d] %02d"%(frame_count,card_list[0]))
                print(matching_text)
                
            if len(card_list) > 1:
                #print(card_list)
                matching_text = ("[%04d]"%(frame_count))
                for card_pair in itertools.combinations(card_list,2):
                    #print("\t",card_pair)
                    card1_mapping = mapping[card_pair[0]]
                    card2_mapping = mapping[card_pair[1]]
                    symbol_ids = np.intersect1d(card1_mapping,card2_mapping)
                    #print("\t",symbol_ids)
                    symbol_id = symbol_ids[0]
                    symbol_label = symbols[symbol_id]
                    #print("\t",symbol_id," => ",symbol_label)
                    matching_text = matching_text + (" %02d,%02d=%s"%(card_pair[0],card_pair[1],symbol_label) )
                print(matching_text)
                cv2.putText(output,matching_text,(matching_x,matching_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)                

            # display real-time FPS counter (if valid)
            if rt_fps_valid == True:
                cv2.putText(output,rt_fps_message, (rt_fps_x,rt_fps_y),text_fontType,text_fontSize,text_color,text_lineSize,text_lineType)
            
            # show the output image
            #img1 = np.hstack([image, output])
            #img2 = np.hstack([cv2.merge([gray1,gray1,gray1]), cv2.merge([gray2,gray2,gray2])])
            #cv2.imshow("dobble detection", np.vstack([img1,img2]))
            cv2.imshow("Dobble Classification", output)

    if step == True:
        key = cv2.waitKey(0)
    elif pause == True:
        key = cv2.waitKey(0)
    else:
        key = cv2.waitKey(10)

    #print(key)
    
    if key == 119 or captureAll == True: # 'w'
        for i in range(0,len(card_list)):
            #print("circle_list[",i,"]=",circle_list[i])
            #print("bbox_list[",i,"]=",bbox_list[i])
            #print("card_list[",i,"]=",card_list[i])
            card_id = card_list[i]
            card_title = "card"+str(card_id)
            bbox = bbox_list[i]
            card_img = image[ bbox[1]:bbox[3], bbox[0]:bbox[2] ]
            #cv2.imshow( card_title, card_img )
            #timestr = datetime.now().strftime("%y_%b_%d_%H_%M_%S_%f")
            #filename = timestr+ "_" + card_title + ".tif"
            filename = ("frame%04d_object%d_card%02d.tif"%(frame_count,i,card_id))
            
            print("Capturing ",filename," ...")
            cv2.imwrite(os.path.join(output_dir,filename),card_img)
       
    if key == 115: # 's'
        step = True    
    
    if key == 112: # 'p'
        pause = not pause

    if key == 99: # 'c'
        step = False
        pause = False

    if key == 27:
        break

    # Update the FPS counter
    fps.update()

    # Update the real-time FPS counter
    rt_fps_count = rt_fps_count + 1
    if rt_fps_count == 10:
        t = (cv2.getTickCount() - rt_fps_time)/cv2.getTickFrequency()
        rt_fps_valid = 1
        rt_fps = 10.0/t
        rt_fps_message = "FPS: {0:.2f}".format(rt_fps)
        #print("[INFO] ",rt_fps_message)
        rt_fps_count = 0



# Stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] elapsed FPS: {:.2f}".format(fps.fps()))

# Cleanup
cv2.destroyAllWindows()
