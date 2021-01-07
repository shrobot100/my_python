#!/usr/bin/env python 
import rospy
import rosbag
import rospy
import cv2
import csv
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

timestamps = [];
images = [];
for topic, msg, t in  rosbag.Bag('/mnt/home/bag/ib/1222/2020-12-22-16-12-42.bag').read_messages():
	if topic == '/zed_node/left/image_rect_color/compressed':
		timestamps.append(msg.header.stamp.to_sec())
		#images.append( CvBridge().imgmsg_to_cv2(msg, "8UC3") )
		try:
			np_arr = np.fromstring(msg.data, np.uint8)
			raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		except CvBridgeError, e:
			print e
		print(str(msg.header.stamp.to_sec()))
		images.append(raw_image)
i = 0
for img in images:
	filename = 'zed' + str(i) + '.jpg'
	print(filename)
	cv2.imwrite(filename, img)
	i = i+1

f = open('timestamp.csv','w')
for time in timestamps:
	f.write(hex(time)+'\n')
f.close()
print('finish')
