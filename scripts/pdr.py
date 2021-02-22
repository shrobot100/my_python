#!/usr/bin/env python
import rospy
from sensor_msgs.msg import Imu
from sensor_msgs.msg import MagneticField
import sys
import copy
import datetime
import pandas as pd
import numpy as np
import tf
import math

acc_list = []

class Node:
	
	def MagneticFieldCallback(self,data):
		print('callback mag')
	def ImuCallback(self,data):
		accel_offset = np.array([0.04002,-0.08126,-0.000596])
		accel_raw = np.array([data.linear_acceleration.x,data.linear_acceleration.y,data.linear_acceleration.z])
		accel_calib = accel_raw - accel_offset
		print('accel calibrated')
		print(accel_calib)
		#acc_list.append(copy.deepcopy(accel_raw))
		#if len(acc_list)>100:
	#			del acc_list[0]
	#	print('average acc')
	#	acc_sum = np.zeros(3)
	#	for i in range(len(acc_list)):
	#		acc_sum += acc_list[i]
	#	print(acc_sum/len(acc_list))
		rpy = np.zeros(3)
		rpy[0] = math.atan2(accel_calib[1],accel_calib[2]) #tan(ay/az)
		rpy[1] = math.atan(-accel_calib[0]/math.sqrt(accel_calib[1]**2+accel_calib[2]**2))
		rpy[2] = 0.0
		print('accel_raw')
		print(accel_raw)
		print('roll')
		print(math.degrees(rpy[0]))
		print('pitch')
		print(math.degrees(rpy[1]))
		mat = self.getRotationMatrix(rpy[0],rpy[1],rpy[2])
		accel_world = np.dot(mat,accel_calib.T)
		print('accel world')
		print(accel_world)

	def getRotationMatrix(self,roll,pitch,yaw):
		mat = np.zeros((3,3))
		rot_x = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
		rot_y = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
		rot_z = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
		return np.dot(rot_z,np.dot(rot_y,rot_x))

	def __init__(self):
		rospy.init_node('pdr',anonymous=True)
		rospy.Subscriber("/android/tango1/imu",Imu,self.ImuCallback)
		#rospy.Subscriber("/android/tango1/magnetic_field",MagneticField,self.MagneticFieldCallback)
	def run(self):
		print('node is running')
		rospy.spin()
def main():
	node = Node()
	node.run()



if __name__ == "__main__":
	main()

