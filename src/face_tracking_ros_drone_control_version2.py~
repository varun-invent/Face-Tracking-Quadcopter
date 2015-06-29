#!/usr/bin/env python

import roslib; roslib.load_manifest('image_ardrone')
import rospy
import sys
import cv2
import cv2.cv as cv
import time,math
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import dlib
from skimage import io

# Import the messages we're interested in sending and receiving
from geometry_msgs.msg import Twist      # for sending commands to the drone
from std_msgs.msg import Empty           # for land/takeoff/emergency
from ardrone_autonomy.msg import Navdata # for receiving navdata feedback


############################################ Global params ################################################
alti = 0

fil = open("log.txt",'a')

fw = 640 #frame width
fh = 360 #frame height
xo = 320 #object x
yo = 180 #object y
face_to_track =  None
face_counter = 0
prev_face_xy = [0,0]
threshold = 30
FACE_RECOGNITION = 1
FACE_TRACKING = 0
land = False
in_air = -1
features0 = None
old_gray = None




############################## Cascade Classifiers for face and eye detection ##############################
face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')

#eye_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_mcs_eyepair_small.xml')




################################## Parameters for lucas kanade optical flow  ###########################
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#####################################################################

pubLand=None

def goodFeatures(img,x,y):
    #print "face roi image shape",img.shape
    img_height = int(img.shape[0]*.05)
    img_width = int(img.shape[1]*.05)
    face_mid_x,face_mid_y = int(img.shape[0]/2),int(img.shape[1]/2)
    features = []
    for i in range(-1*img_height,img_height,3):
        for j in range(-1*img_width,img_width,3):
            features.append([y+face_mid_y+j,x+face_mid_x+i])
    return features
    
def image_callback(ros_image):
	#print "In image_callback"
	fil.write("In image_callback\n")
	bridge = CvBridge()
	node_name = 'disp image'
	try:
		frame = bridge.imgmsg_to_cv2(ros_image, "bgr8")
	except CvBridgeError, e:
		print e
    
	frame = np.array(frame, dtype=np.uint8)  # Converting IPL Image to np array
	frame_size = np.shape(frame)
    
	fw,fh = frame_size[1],frame_size[0]
	x_mid,y_mid = fw/2,fh/2
	cv2.circle(frame,(x_mid,y_mid), 15, (80,80,80,100), -1)  # Drawing circle at the centre of the frame
	
    ###########################################################
	global face_cascade
	global eye_cascade
	global FACE_RECOGNITION
	global FACE_TRACKING
	
	
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	if FACE_RECOGNITION == 1:
		faces = face_cascade.detectMultiScale(gray_frame,1.2,5)
		print "Found {0} faces".format(len(faces))
		if len(faces)>0:
			max_face = faces[0] # max_face is the face to track
			max_diagonal = 0
			for i,(x,y,w,h) in enumerate(faces):   # Finds the nearest face
				cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
				current_diagonal = np.sqrt(w**2 +h**2)
				
				##### Eye Detect###
				#roi_gray = gray_frame[y:y+h, x:x+w]
				#roi_color = frame[y:y+h, x:x+w]
				#eyes = eye_cascade.detectMultiScale(roi_gray,1.2,4)
				#print "Found {0} eyes".format(len(eyes))
				#for (ex,ey,ew,eh) in eyes:
					#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
				####################
				  
				if current_diagonal >= max_diagonal: # checking the largest face bbox
					max_face =  faces[i]
					max_diagonal = current_diagonal 
				
				global prev_face_xy       
				print "abs(prev_face_xy[0]-max_face[0])",abs(prev_face_xy[0]-max_face[0])
				if abs(prev_face_xy[0]-max_face[0])<threshold:   # checks if the currently detected face is close to the previous detected face
					 global face_counter                         # So as to get rid of false negatives   
					 face_counter = face_counter + 1
				
				prev_face_xy = max_face[:2]
				
				#Print all the faces, False Negatives and eyes
				cv2.imshow('Detected Face, False Negatives and Eyes',frame)
				cv2.waitKey(5) 
			
			##################### Now fixing the Face #####  #################    
				
			if face_counter == 2:                       # Check is the same(or nearby) frame is detected 3 consequtive times
				face_to_track = max_face                 # To get rid of false negatives
				x= face_to_track[0]
				y= face_to_track[1]
				w= face_to_track[2]
				h= face_to_track[3]
				cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2) # Face ROI
			   
				##### Eye Detect###
				#roi_gray = gray_frame[y:y+h, x:x+w]
				#roi_color = frame[y:y+h, x:x+w]
				#eyes = eye_cascade.detectMultiScale(roi_gray,1.2,4)
				#print "Found {0} eyes".format(len(eyes))
				#for (ex,ey,ew,eh) in eyes:
					#cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)       # Eye ROI         
				
				face_counter = 0 # After fixing the face, set the face_counter to zero
				 
				#if len(eyes) > 0:
					#(ex,ey,ew,eh) = eyes[0]
					#print "Red Eyes xywh",ex,ey,ew,eh
					#print "printing eyes"
					#cv2.imshow('Img_eye_Tracked',frame[y+ey:y+ey+eh,x+ex:x+ex+ew])
					#cv2.waitKey(1)
				global features0
					#features0 = goodFeatures(frame[y+ey:y+ey+eh,x+ex:x+ex+ew],(y+ey),(x+ex)) # sends the eyes
				features0 = goodFeatures(frame[y:y+h, x:x+w],y,x)
				FACE_RECOGNITION = 0
				FACE_TRACKING = 1
				global old_gray
				old_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
				#cv2.imshow("Fixed Face and Eyes",frame)
				#cv2.waitKey(5)    
					
					
	if FACE_TRACKING ==1:
		frame_tracking = frame.copy() # Frame to display the tracked points
		
		features0 = np.float32(np.array(features0)) # converting the features to LKT compaltible type
		
		
		# calculate optical flow and get the new location of tracked pixel##
		features1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, features0, None, **lk_params)
		st = np.array(st).ravel()
		
		global xo,yo 
		if features1 == None:
			FACE_TRACKING = 0 # Turn off tracking and turn on recognition of features are lost
			FACE_RECOGNITION = 1 
			xo,yo = -1,-1
			return
			
		good_new = features1[st==1]  # Selecting the piexls tracked in the next frame
		good_new = np.array(good_new) # converting to compaltible type so that it could be enumerated in a loop
			
		for (px,py) in good_new:
			if int(py) in range(480) and int(px) in range(640):
				frame_tracking[int(py),int(px)]= (0,255,0)                  # Setting the tracked points
		
		#cv2.imshow("Tracked Features",frame_tracking)
		#cv2.waitKey(5)
		
		features0 = good_new
		old_gray = gray_frame
		
		[mean_x,mean_y] = np.mean(good_new,axis=0) # finding the mean of the cluster of tracked points
		
		   
		xo,yo = mean_x,mean_y
		
		cv2.circle(frame_tracking,(mean_x,mean_y), 6, (80,80,80,100), -1)  # Drawing circle at the centre of the frame
		cv2.imshow("Tracked Features",frame_tracking)
		cv2.waitKey(5)
		
		
					
    ###########################################################
    
      
   	
	#cv2.imshow(node_name, frame)


	keystroke = cv.WaitKey(5)
	if 32 <= keystroke and keystroke < 128:
		cc = chr(keystroke).lower()
		if cc == 'q':
			land == True	
			print "pressed q"		
			#print "Before WaitKey"
			#if cv2.waitKey(5) == 27:
			#global pubLand
			#pubLand.publish(Empty())
			#rospy.signal_shutdown("User hit q key to quit.")
	if land== True:
		global pubLand
		pubLand.publish(Empty())
		rospy.signal_shutdown("User hit q key to quit.")
		
	global alti
	print "Altitude is",alti
	fil.write("Altitude is"+str(alti)+"\n")
	
def cleanup():
	print "Shutting down vision node."
	cv2.destroyAllWindows()   

def nav_callback(navdata):
	global alti
	alti = navdata.altd

def face_drone():
	global pubLand
	#print "In face_drone"
	fil.write("In face_drone\n")
	pubLand=rospy.Publisher('/ardrone/land',Empty,queue_size=10)
	pubReset   = rospy.Publisher('/ardrone/reset',Empty,queue_size=10)
	pubTakeoff= rospy.Publisher('/ardrone/takeoff',Empty,queue_size=10)
	pubCommand = rospy.Publisher('/cmd_vel',Twist,queue_size=10)
	
	rospy.Subscriber("/ardrone/image_raw", Image, image_callback)
	rospy.Subscriber("/ardrone/navdata",Navdata,nav_callback)
	rospy.init_node('face_drone')
	
	global fw,fh,xo,yo
	if fw > 0:
		#print "In face_drone"
		fil.write("In face_drone\n")
		fwm = fw/2
		fhm = fh/2
		oxm = xo
		oym = yo
		#diag = math.sqrt(wo*wo+ho*ho)
		#print "oxm,oym",oxm,oym
		r = rospy.Rate(10)  # 10 Hz
		empty = Empty()
		twist = Twist()
		
		#pubTakeoff.publish(empty)
		#rospy.sleep(5)
		#twist.linear.x = 0
		#twist.linear.y = 0
		#twist.linear.z = 0.5
		#twist.angular.z = 0
		#pubCommand.publish(twist)
		#rospy.sleep(2)
		global in_air
		ex_max = fwm
		while not rospy.is_shutdown():
			#pubTakeoff.publish(empty)                              # Togge to takeoff
			print "published command to takeoff in while loop"
			fil.write("publish to takeoff in while loop\n")
			if alti > 5 and in_air == -1:
				#print "in air but not rising"
				rospy.sleep(6)
				fil.write("in air but not rising")
				in_air  = 1
				
			if in_air == 1:
				#pubTakeoff.publish(empty)
				#rospy.sleep(2.0)
				#print "in air and rising"
				fil.write("in air and rising\n")
				twist.linear.x = 0
				twist.linear.y = 0
				twist.linear.z = 0.1
				twist.angular.z = 0
				#print "publshing altitude"
				fil.write("publishing altitude\n")
				pubCommand.publish(twist)
				#print "published altitude"
				fil.write("published altitude\n")
				rospy.sleep(3.0)
				print "rose by now"
				fil.write("rose by now\n")
				in_air = 0
			#spubTakeoff.publish(empty)
			#hover
			#global fw,fh,xo,yo,wo,ho
			fwm = fw/2
			fhm = fh/2
			if xo == -1:
			    ex=0
			else:	
				oxm = xo
				oym = yo
				ex = (0.45)*(fwm-oxm)/float(ex_max)
			#diag = math.sqrt(wo*wo+ho*ho)
			#print "oxm,oym",oxm,oym
			print "Err,fwm,oxm,ex_max",ex,fwm,oxm,ex_max
			fil.write("Err,fwm,oxm,ex_max\n"+str(ex)+str(fwm)+str(oxm)+str(ex_max))
			twist.linear.x = 0
			twist.linear.y = 0
			twist.linear.z = 0
			twist.angular.x = 0
			twist.angular.y = 0
			twist.angular.z = ex
			pubCommand.publish(twist)
			
		rospy.sleep(0.9)
	if land== True:
		pubLand.publish(Empty())
		rospy.signal_shutdown("User hit q key to quit.")

def corners(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
    corners = np.int0(corners)

    #for i in corners:
    #    x,y = i.ravel()			

if __name__ == '__main__':
	try:
		face_drone()
	except rospy.ROSInterruptException: 
		pass
fil.close()

