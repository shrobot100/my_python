#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
from std_srvs.srv import Empty
from std_srvs.srv import EmptyResponse
from std_msgs.msg import Header

import os
import sys
import time
import copy
import datetime
import pandas as pd
import numpy as np
import tf
import math

buf_x = []
buf_y = []
buf_z = []
def main():
	rospy.init_node('node')
	rospy.Subscriber('/android/tango1/imu',Imu,callback)
	rospy.spin()

def callback(data):
	buf_x.append(data.linear_acceleration.x)
	buf_y.append(data.linear_acceleration.y)
	buf_z.append(data.linear_acceleration.z)

	if len(buf_x)==2000:
		print('x:',sum(buf_x)/len(buf_x))
		print('y:',sum(buf_y)/len(buf_y))
		print('z:',sum(buf_z)/len(buf_z))
		buf_x.clear()
		buf_y.clear()
		buf_z.clear()



if __name__ == "__main__":
	main()

