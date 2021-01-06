import rosbag
import rospy
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

timestamps = [];
images = [];
for topic, msg, t in  rosbag.Bag('HOGEFUGA.bag').read_messages():
	if topic == '/zed/left/image_raw_color':
		timestamps.append(t.to_sec())
		images.append( CvBridge().imgmsg_to_cv2(msg, "8UC3") )
