import numpy as np
import cv2

cv2.namedWindow('frame')

# return contours
def find_totem(frame, lower_hsv, upper_hsv):
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
    res = cv2.bitwise_and(frame, frame, mask= mask)
    
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edged = cv2.Canny(gray, 35, 125)
    
    kernel = np.ones((1,1),np.uint8)
    erosion = cv2.erode(edged,kernel,iterations = 1)
    
    kernel = np.ones((2,2),np.uint8)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    
    (contours, _) = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    ## Find the index of the largest contour
    areas = [cv2.contourArea(c) for c in contours]
    if (areas):
        max_index = np.argmax(areas)
        cnt=contours[max_index]
        
        return cv2.boundingRect(cnt)
        
        #x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        #total = 0;
        #i = 0
        # loop over the contours
        #for c in contours:
        #    #print i
        #    # approximate the contour
        #    peri = cv2.arcLength(c, True)
        #    #print peri
        #    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        #    
        #    if len(approx) == 4:
        #        return approx
            
    return 0


#cap = cv2.VideoCapture(0)

# the video format might be an issue if using recorded data, direct camera input might work well
#cap = cv2.VideoCapture('/Users/miguelnunes/GoogleDrive/RobotX/video/videoplayback.mp4')
cap = cv2.VideoCapture('video/test_totem_1.mov')
#cap = cv2.VideoCapture('E:/GoogleDrive/RobotX/video/test15.mov')



def nothing(x):
    pass
    
#hsv_green_lower = np.array([65, 27, 0])
#hsv_green_upper = np.array([72,255,255]) 
    
#cv2.createTrackbar('h_min', 'frame',65,179,nothing)
#cv2.createTrackbar('s_min', 'frame',50,255,nothing)
#cv2.createTrackbar('v_min', 'frame',0,255,nothing)
#
#cv2.createTrackbar('h_max', 'frame',83,179,nothing)
#cv2.createTrackbar('s_max', 'frame',255,255,nothing)
#cv2.createTrackbar('v_max', 'frame',255,255,nothing)

while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
#    #get info from track bar and appy to frame
#    h_min = cv2.getTrackbarPos('h_min','frame')
#    s_min = cv2.getTrackbarPos('s_min','frame')
#    v_min = cv2.getTrackbarPos('v_min','frame')
#    
#    print 'hsv_min = ', h_min, s_min,v_min
#
#    h_max = cv2.getTrackbarPos('h_max','frame')
#    s_max = cv2.getTrackbarPos('s_max','frame')
#    v_max = cv2.getTrackbarPos('v_max','frame')
#    
#    print 'hsv_max = ', h_max, s_max,v_max
#    
#    # green HSV
#    #hsv_green_lower = np.array([10, 100, 10])
#    #hsv_green_upper = np.array([100,255,255]) 
#    
#    # green mit
#    hsv_green_lower = np.array([h_min, s_min, v_min])
#    hsv_green_upper = np.array([h_max, s_max, v_max]) 
    
    #hsv_green_lower = np.array([65, 50, 0])
    #hsv_green_upper = np.array([83,255,255]) 
    
    hsv_green_lower = np.array([51, 30, 0])
    hsv_green_upper = np.array([64,255,255]) 
    
    hsv_green_lower = np.array([60, 50, 0])
    hsv_green_upper = np.array([74,255,255]) 
    
    # red HSV
    hsv_red_lower = np.array([160, 100, 0])
    hsv_red_upper = np.array([180,255,255]) 
    
    contour_red = find_totem(frame, hsv_red_lower, hsv_red_upper)
    contour_green = find_totem(frame, hsv_green_lower, hsv_green_upper)
    
    if contour_red:
        x,y,w,h = contour_red
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    
    if contour_green:
        x,y,w,h = contour_green
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
 

    # Display the resulting frame
    #cv2.imshow('frame',frame)
    cv2.imshow('frame',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()