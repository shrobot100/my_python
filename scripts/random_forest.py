#!/usr/bin/env python
import rosbag
import rospy
import cv2
import sys
import copy
import pandas as pd
import numpy as np
import tf
import random
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

class Pose_C:
	def __init__(self,timestamp,pos,orientation):
		self.__timestamp = timestamp
		self.__pos = pos
		self.__orientation = orientation
	def getTime(self):
		return self.__timestamp
	def getPos(self):
		return self.__pos
	def getOrientation(self):
		return self.__orientation
	def setPos(self,in_pos):
		self.__pos = in_pos
	def setOrientation(self,in_ori):
		self.__orientation = in_ori
	def info(self):
		print(self.__timestamp)
		print(self.__pos)
		print(self.__orientation)

	def getRelativePose(self,origin_pose):
		origin_quat = origin_pose.getOrientation()

		origin_quat_inv = copy.deepcopy(origin_quat)
		origin_quat_inv[3] = -origin_quat_inv[3]

		origin_pos = np.zeros(4)
		origin_pos[0:3] = origin_pose.getPos()

		raw_pos = np.zeros(4)
		raw_quat = self.__orientation
		raw_pos[0:3] = self.__pos


		relative_pos = raw_pos - origin_pos

		tmp = tf.transformations.quaternion_multiply(origin_quat,relative_pos)
		new_relative_pos = tf.transformations.quaternion_multiply(tmp,origin_quat_inv)
#TODO
		new_relative_quat = tf.transformations.quaternion_multiply(raw_quat,origin_quat_inv)
		
		ret_pose = Pose_C(self.__timestamp,new_relative_pos[0:3],new_relative_quat)
		return ret_pose
		

class Odometry_C:
	def __init__(self):
		self.array = []
		self.name = ''
	def size(self):
		return len(self.array)
	def getName(self):
	 return self.name
	def load(self,filename):
		self.name = filename
		print('loading csv: ' + filename)
		df = pd.read_csv(filename)
		for i in range(len(df)):
			pos = np.zeros(3)
			orientation = np.zeros(4)
			stamp_num = df['field.header.stamp'][i]
			stamp_str = str(stamp_num)
			stamp_float = float(stamp_str[0:10] + '.' + stamp_str[10:])
			pos[0] = df['field.pose.pose.position.x'][i]
			pos[1] = df['field.pose.pose.position.y'][i]
			pos[2] = df['field.pose.pose.position.z'][i]

			orientation[0] = df['field.pose.pose.orientation.x'][i]
			orientation[1] = df['field.pose.pose.orientation.y'][i]
			orientation[2] = df['field.pose.pose.orientation.z'][i]
			orientation[3] = df['field.pose.pose.orientation.w'][i]

			pose = Pose_C(stamp_float,pos,orientation)
			self.array.append(copy.deepcopy(pose))

		print('odom size:' + str(len(self.array)))
	
	def extract(self,init_time,th_time):
		new_array = []
		min_delta_list = []
		end_time = self.array[-1].getTime()
		time_list = np.arange(init_time,end_time,th_time)

		for time_cnt in time_list:
			min_delta = sys.float_info.max
			for i in range(len(self.array)):
				delta = abs(self.array[i].getTime() - time_cnt)
				if min_delta>=delta:
					min_delta = delta
				else:
					new_array.append(self.array[i-1])
					break
			min_delta_list.append(min_delta)
		
		print('max_delta:' + str(max(min_delta_list)))
		self.array = new_array
	def extractFromTimeList(self,time_list):
		new_array = []
		min_delta_list = []
		end_time = self.array[-1].getTime()

		for time_cnt in time_list:
			if time_cnt > end_time:
				break
			min_delta = sys.float_info.max
			for i in range(len(self.array)):
				delta = abs(self.array[i].getTime() - time_cnt)
				if min_delta>=delta:
					min_delta = delta
				else:
					new_array.append(self.array[i-1])
					break
			min_delta_list.append(min_delta)
		
		print('max_delta:' + str(max(min_delta_list)))
		print('max_delta_index' + str(min_delta_list.index(max(min_delta_list))))
		print('size:' + str(len(self.array)) + '->' + str(len(new_array)))
		self.array = new_array
	
	
	def getTimeStamps(self):
		ret_array = []
		for pose in self.array:
			ret_array.append(pose.getTime())

		return ret_array

	def setScale(self,scale):
		for pose in self.array:
			raw_pos = pose.getPos()
			pose.setPos(raw_pos*scale)	

	def rotate(self,roll,pitch,yaw):
		for pose in self.array:
			q_rot = tf.transformations.quaternion_from_euler(roll,pitch,yaw)
			q_rot_inv =	tf.transformations.quaternion_inverse(q_rot)
			q_new = tf.transformations.quaternion_multiply(q_rot,pose.getOrientation())
			pose.setOrientation(q_new)
			q_pos = np.zeros(4)
			q_pos[0:3] = pose.getPos()
			q_new_pos_tmp = tf.transformations.quaternion_multiply(q_rot,q_pos)
			q_new_pos = tf.transformations.quaternion_multiply(q_new_pos_tmp,q_rot_inv)
			pose.setPos(q_new_pos[0:3])


	def getRelativeOdom(self):
		new_array = []
		new_array.append(copy.deepcopy(self.array[0]))
		for i in range(1,len(self.array)):
			pose = self.array[i].getRelativePose(self.array[i-1])
			new_array.append(copy.deepcopy(pose))
		
		ret = Odometry_C()
		ret.array = new_array
		return ret

	def plot(self):
			
		x_list = []
		y_list = [] 
		z_list = []
		for tmp in self.array:
			x_list.append(tmp.getPos()[0])
			y_list.append(tmp.getPos()[1])
			z_list.append(tmp.getPos()[2])
		ax.plot(x_list,y_list,label = self.name)
		

#TODO
	def transform2AbusoluteOdom(self):
		new_array = []
		new_array.append(copy.deepcopy(self.array[0]))
		for i in range(1,len(self.array)):
			r_pos = self.array[i].getPos()
			r_ori = self.array[i].getOrientation()
			pre_pose = self.array[i].getPos()
			pre_ori = self.array[i].getOrientation

			#pos =	 
	def length(self):
		length = 0.0
		for  i in range(1,len(self.array)):
			delta = self.array[i].getPos() - self.array[i-1].getPos()
			length = length + np.linalg.norm(delta)

		return length

class Images_C:
	def __init__(self):
		self.timestamps = []	
		self.images = []
	def load(self,filepath,topic_name):
		for topic, msg, t in  rosbag.Bag(filepath).read_messages():
			if topic == topic_name:
				self.timestamps.append(msg.header.stamp.to_sec())
				#images.append( CvBridge().imgmsg_to_cv2(msg, "8UC3") )
				#try:
				np_arr = np.fromstring(msg.data, np.uint8)
				raw_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
				#except CvBridgeError, e:
				#	print e
				print(str(msg.header.stamp.to_sec()))
				self.images.append(raw_image)

	
	def extract(self,init_time,th_time):
		new_array = []
		min_delta_list = []
		end_time = self.timestamps[-1]
		time_list = np.arange(init_time,end_time,th_time)

		for time_cnt in time_list:
			min_delta = sys.float_info.max
			for i in range(len(self.timestamps)):
				delta = abs(self.timestamps[i] - time_cnt)
				if min_delta>=delta:
					min_delta = delta
				else:
					new_array.append(self.timestamps[i-1])
					break
			min_delta_list.append(min_delta)
		
		print('max_delta:' + str(max(min_delta_list)))
		self.images = new_array
		
def main():
	ref_odom = Odometry_C()
	ref_odom.load('lio_loop.csv')
	print('ref_odom length:' + str(ref_odom.length()))
	
	openvslam_odom = Odometry_C()
	openvslam_odom.load('./loop/openvslam_loopmap_localize.csv')
	openvslam_odom.setScale(ref_odom.length() / openvslam_odom.length())
	print('openvslam_map_length' + str(openvslam_odom.length()))
	openvslam_odom.rotate(0.0,0.0,0.05)

	orb_zed_odom = Odometry_C()
	orb_zed_odom.load('./loop/orb_zed_loop_localize.csv')
	orb_zed_odom.setScale(ref_odom.length() / orb_zed_odom.length())
	orb_zed_odom.rotate(0.0,0.0,0.05)

	'''
	orb_rs_odom = Odometry_C()
	orb_rs_odom.load('./loop/orb_rs_loop_localize.csv')
	orb_rs_odom.setScale(ref_odom.length() / orb_rs_odom.length())
	orb_rs_odom.rotate(0.0,0.0,0.05)
	'''


	ref_timestamps = ref_odom.getTimeStamps()

	openvslam_odom.extractFromTimeList(ref_timestamps)
	orb_zed_odom.extractFromTimeList(ref_timestamps)

	openvslam_odom.plot()
	orb_zed_odom.plot()
	#orb_rs_odom.plot()
	ref_odom.plot()

	#create dataset
	sample = []
	min_size = min(ref_odom.size(),openvslam_odom.size(),orb_zed_odom.size())

	for i in range(min_size):
		#x = np.hstack(openvslam_odom.array[i].getPos(), orb_zed_odom.array[i].getPos())
		x = (openvslam_odom.array[i].getPos(),orb_zed_odom.array[i].getPos())
		sample.append((np.hstack(x),ref_odom.array[i].getPos()))

	#create training data,test data
	boundary = 600
	#r_sample = random.sample(sample,len(sample))	
	r_sample = sample

	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i in range(0,boundary):
		x_train.append(r_sample[i][0])
		y_train.append(r_sample[i][1])
	
	for i in range(boundary,min_size):
		x_test.append(r_sample[i][0])
		y_test.append(r_sample[i][1])
	#machine learning
	model = RandomForestRegressor(n_estimators=100,max_depth=100)
	model.fit(x_train,y_train)
	importance = model.feature_importances_   
# 学習データの正解率
	print("Train :", model.score(x_train, y_train))

# テストデータの正解率
	print("Test :", model.score(x_test, y_test))
	
	y_train_pred = model.predict(x_train)
	y_pred = model.predict(x_test)
	print('MSE train data: ', mean_squared_error(y_train, y_train_pred)) # 学習データを用いたときの平均二乗誤差を出力
	print('MSE test data: ', mean_squared_error(y_test, y_pred))         # 検証データを用いたときの平均二乗誤差を出力

#重要度
	for num in importance:
		print(num)

	predict_input = []
	for i in range(min_size):
		predict_input.append(sample[i][0])

	output = model.predict(predict_input)
	ax.plot(output[:,0],output[:,1],label = 'prediction')
	print(sample[boundary][1][0])
	print(sample[boundary][1][1])
	ax.scatter([sample[boundary][1][0]],[sample[boundary][1][1]],color='blue' ,zorder=100)
	ax.legend()
	plt.savefig('not_random.png')
	
fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(111)
if __name__ == "__main__":
	main()


