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


acc_calib_buf = []
acc_calib_buf_size = 100
class Node:
	
	def __init__(self):
		rospy.init_node('pdr',anonymous=True)
#Subscriber
		rospy.Subscriber("/android/tango1/imu",Imu,self.ImuCallback)
#Publisher
		self.acc_pub = rospy.Publisher('/accel_world',Imu,queue_size=10)
		self.step_detection_pub = rospy.Publisher('/step_detection',Header,queue_size=10)
#Service
		rotateSrv = rospy.Service('rotation',Empty,self.handleRotation)
		
		#rospy.Subscriber("/android/tango1/magnetic_field",MagneticField,self.MagneticFieldCallback)
		self.rate = rospy.Rate(10)
		self.pre_gravity_z = 9.8
		
		self.rot_matrix= getRotationMatrix(0.0,0.0,0.0)
		self.footdetection = footDetection()
	def handleRotation(self,req):
		print('calculate rotation matrix')
		if len(acc_calib_buf)<5:
			print('error')
			return EmptyResponse()
		tmp_acc = (acc_calib_buf[-1]+acc_calib_buf[-2]+acc_calib_buf[-3]+acc_calib_buf[-4]+acc_calib_buf[-5])/5
		rpy = np.zeros(3)
		rpy[0] = math.atan2(tmp_acc[1],tmp_acc[2]) #tan(ay/az)
		rpy[1] = math.atan(-tmp_acc[0]/math.sqrt(tmp_acc[1]**2+tmp_acc[2]**2))
		rpy[2] = 0.0
		print('roll')
		print(math.degrees(rpy[0]))
		print('pitch')
		print(math.degrees(rpy[1]))
		self.rot_matrix= getRotationMatrix(rpy[0],rpy[1],rpy[2])
		return EmptyResponse()

	def MagneticFieldCallback(self,data):
		print('callback mag')
	def ImuCallback(self,data):
		accel_offset = np.array([0.04002,-0.08126,-0.000596])
		accel_raw = np.array([data.linear_acceleration.x,data.linear_acceleration.y,data.linear_acceleration.z])
		accel_calib = accel_raw - accel_offset
		acc_calib_buf.append(accel_calib)
		if len(acc_calib_buf)>acc_calib_buf_size:
			del acc_calib_buf[0]
		accel_gcs = np.dot(self.rot_matrix,accel_calib.T)

		flag = self.footdetection.run(accel_gcs[2],10)
		#if flag==True:
			#foot_step_timestamp = data.header.timestamp
			#pre_foot_step_timestamp = foot_step_timestamp
		#print(flag)	

		"""
		imu_world_out = Imu()
		imu_world_out.header = data.header
		imu_world_out.linear_acceleration.x = 0.0
		imu_world_out.linear_acceleration.y = accel_hpf 
		imu_world_out.linear_acceleration.z = accel_step

		self.acc_pub.publish(imu_world_out)
		"""
		


	def run(self):
		print('node is running')
		while not rospy.is_shutdown():
			self.rate.sleep()


class footDetection:
	def __init__(self):
		self.acc_step_buf_size = 50
		self.acc_step_buf = []
		self.pre_gravity_z = 9.8
	def run(self,accel_z,winsize):
		accel_step = calStepAccel(accel_z,10)
		if accel_step != None:
			self.acc_step_buf.append(accel_step)
			if len(self.acc_step_buf)>self.acc_step_buf_size:
				del self.acc_step_buf[0]
				min_index = self.acc_step_buf.index(min(self.acc_step_buf)) #サンプルの中で最小値のインデックスを得る
				if min_index==self.acc_step_buf_size//2 and self.acc_step_buf[self.acc_step_buf_size//2]< -1.0:
					rospy.loginfo('detect step')	
					return True
		return False
	
class footLengthEstimator:
	def __init__(self):
		alpha = 0.0	
		beta = 0.0
	def run(self):
		acc_step_max = 0.0
		acc_step_min = 0.0
		acc_pp = 0.0
		length = alpha*(acc_pp ** (1/4))+beta

def calStepAccel(accel_z,winsize):
		#ローパスフィルタで重力成分抽出
		gravity_z = lowPassFilter(accel_z,calStepAccel.pre_gravity_z,0.9)
		calStepAccel.pre_gravity_z = gravity_z
		
		#重力成分以外を抽出
		accel_hpf = accel_z - gravity_z
		calStepAccel.acc_hpf_buf.append(accel_hpf)
		if len(calStepAccel.acc_hpf_buf)>calStepAccel.acc_hpf_buf_size:
			del calStepAccel.acc_hpf_buf[0]
			accel_step = calMovingAverage(calStepAccel.acc_hpf_buf,winsize)
			return accel_step
calStepAccel.pre_gravity_z = 9.8
calStepAccel.acc_hpf_buf = []
calStepAccel.acc_hpf_buf_size= 100

def getRotationMatrix(roll,pitch,yaw):
	mat = np.zeros((3,3))
	rot_x = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
	rot_y = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
	rot_z = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
	return np.dot(rot_z,np.dot(rot_y,rot_x))
def lowPassFilter(data,pre_data,alpha):
	ret = alpha*pre_data + (1-alpha)*data
	return ret
def calMovingAverage(data,win_size):
	sum_var = 0.0
	for i in range(win_size):
		sum_var+=data[-i-1]
	return sum_var/win_size


def main():
	try:
		node = Node()
		node.run()
	except rospy.ROSInterruptException: pass



if __name__ == "__main__":
	main()

