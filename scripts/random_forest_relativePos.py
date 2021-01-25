#!/usr/bin/env python
import rosbag
import rospy
import cv2
import sys
import copy
import datetime
import pandas as pd
import numpy as np
import tf
import random
import math
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
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
		origin_quat_inv = tf.transformations.quaternion_inverse(origin_quat) 

		origin_pos = np.zeros(4)
		origin_pos[0:3] = origin_pose.getPos()

		raw_pos = np.zeros(4)
		raw_quat = self.__orientation
		raw_pos[0:3] = self.__pos


		relative_pos = raw_pos - origin_pos

		tmp = tf.transformations.quaternion_multiply(origin_quat_inv,relative_pos)
		new_relative_pos = tf.transformations.quaternion_multiply(tmp,origin_quat)

		new_relative_quat = tf.transformations.quaternion_multiply(raw_quat,origin_quat_inv)
		
		ret_pose = Pose_C(self.__timestamp,new_relative_pos[0:3],new_relative_quat)
		return ret_pose
		

	def getRelativePoseInv(self,origin_pose):
		origin_quat = origin_pose.getOrientation()
		origin_quat_inv = tf.transformations.quaternion_inverse(origin_quat) 

		origin_pos = np.zeros(4)
		origin_pos[0:3] = origin_pose.getPos()

		raw_pos = np.zeros(4)
		raw_quat = self.__orientation
		raw_pos[0:3] = self.__pos

		#Rotation
		q_new_ori = tf.transformations.quaternion_multiply(raw_quat,origin_quat)
		tmp = tf.transformations.quaternion_multiply(origin_quat,raw_pos)
		tmp2 = tf.transformations.quaternion_multiply(tmp,origin_quat_inv)
		#並進
		q_new_pos = tmp2 + origin_pos

		ret_pose = Pose_C(self.__timestamp,q_new_pos[0:3],q_new_ori)
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

	def getRelativeOdomInv(self):
		new_array = []
		new_array.append(copy.deepcopy(self.array[0]))
		for i in range(1,len(self.array)):
			pose = self.array[i].getRelativePoseInv(new_array[i-1])
			new_array.append(copy.deepcopy(pose))
		
		ret = Odometry_C()
		ret.name = self.name
		ret.array = new_array
		return ret

	def plot(self,fignum):
		x_list = []
		y_list = [] 
		z_list = []
		for tmp in self.array:
			x_list.append(tmp.getPos()[0])
			y_list.append(tmp.getPos()[1])
			z_list.append(tmp.getPos()[2])
		if fignum==0:
			ax.plot(x_list,y_list,label = self.name)
		else:
			ax2.plot(x_list,y_list,label = self.name)
	
	def length(self):
		length = 0.0
		for  i in range(1,len(self.array)):
			delta = self.array[i].getPos() - self.array[i-1].getPos()
			length = length + np.linalg.norm(delta)

		return length
	
	def split(self,begin,end):
		new_array = []
		for i in range(begin,end):
			new_array.append(self.array[i])

		new_odom = Odometry_C()
		new_odom.array = new_array
		new_odom.name = self.name
		return new_odom



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

	#openvslam_odom.plot()
	#orb_zed_odom.plot()
	#ref_odom.plot()

	ref_train_odom_abs = ref_odom.split(0,850)
	ref_train_odom_abs.name = 'Lio mapping'
	ref_train_odom_abs.plot(0)

	ref_test_odom_abs = ref_odom.split(850,ref_odom.size())
	ref_test_odom_abs.name = 'Lio mapping'
	ref_test_odom_abs.plot(1)

	open_train_odom_abs = openvslam_odom.split(0,850)
	open_train_odom_abs.name = 'OpenVSLAM(train)'
	open_train_odom_abs.plot(0)

	open_test_odom_abs = openvslam_odom.split(850,openvslam_odom.size())
	open_test_odom_abs.name = 'OpenVSLAM(test)'
	open_test_odom_abs.plot(1)

	orb_train_odom_abs = orb_zed_odom.split(0,850)
	orb_train_odom_abs.name = 'ORB SLAM2(train)'
	orb_train_odom_abs.plot(0)

	orb_test_odom_abs = orb_zed_odom.split(850,orb_zed_odom.size())
	orb_test_odom_abs.name = 'ORB SLAM2(test)'
	orb_test_odom_abs.plot(1)
	#calculate relative pose
	openvslam_odom_rel = openvslam_odom.getRelativeOdom()
	orb_zed_odom_rel = orb_zed_odom.getRelativeOdom()
	ref_odom_rel = ref_odom.getRelativeOdom()
	


	#create dataset
	sample = []
	min_size = min(ref_odom.size(),openvslam_odom.size(),orb_zed_odom.size())

	for i in range(1,min_size):
#過去フレーム参照
		'''
		x = (openvslam_odom_rel.array[i].getPos(),orb_zed_odom_rel.array[i].getPos(),\
		openvslam_odom_rel.array[i-1].getPos(),orb_zed_odom_rel.array[i-1].getPos(),\
		openvslam_odom_rel.array[i-2].getPos(),orb_zed_odom_rel.array[i-2].getPos())
		'''
#現在フレーム参照
		'''
		x = (openvslam_odom_rel.array[i].getPos(),\
		orb_zed_odom_rel.array[i].getPos(),\
		openvslam_odom_rel.array[i].getOrientation(),\
		orb_zed_odom_rel.array[i].getOrientation())
		'''
		#NOTE 順番変更
		x = (openvslam_odom_rel.array[i].getPos(),\
		openvslam_odom_rel.array[i].getOrientation(),\
		orb_zed_odom_rel.array[i].getPos(),\
		orb_zed_odom_rel.array[i].getOrientation())

		y = (ref_odom_rel.array[i].getPos(),ref_odom_rel.array[i].getOrientation())
		sample.append((np.hstack(x),np.hstack(y)))

	#create training data,test data
	#boundary = 500 
	boundary = 850 
	#r_sample = random.sample(sample,len(sample))	
	r_sample = sample

	x_train = []
	y_train = []
	x_test = []
	y_test = []
	for i in range(0,boundary):
		x_train.append(r_sample[i][0])
		y_train.append(r_sample[i][1])
	
	for i in range(boundary,len(r_sample)):
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
	y_test_pred = model.predict(x_test)
	print('MSE train data: ', mean_squared_error(y_train, y_train_pred)) # 学習データを用いたときの平均二乗誤差を出力
	print('MSE test data: ', mean_squared_error(y_test, y_test_pred))         # 検証データを用いたときの平均二乗誤差を出力

#重要度
	for num in importance:
		print(num)
#入力データプロット
		"""
#OpenVSLAM(train)
	OpenVSLAM_train_odom = Odometry_C()
	OpenVSLAM_train_odom.name = 'OpenVSLAM(train)'
	for raw_data in x_train:
		OpenVSLAM_train_odom.array.append(Pose_C(0.0,raw_data[0:3],raw_data[3:7]))
	
	abs_openv_train = OpenVSLAM_train_odom.getRelativeOdomInv()
	abs_openv_train.plot(0)
#ORB SLAM(train)
	orb_train_odom = Odometry_C()
	orb_train_odom.name = 'ORB SLAM2(train)'
	for raw_data in x_train:
		orb_train_odom.array.append(Pose_C(0.0,raw_data[7:10],raw_data[10:]))
	
	abs_orb_train = orb_train_odom.getRelativeOdomInv()
	abs_orb_train.plot(0)
#OpenVSLAM(test)
	OpenVSLAM_test_odom = Odometry_C()
	OpenVSLAM_test_odom.name = 'OpenVSLAM(test)'
	for raw_data in x_test:
		OpenVSLAM_test_odom.array.append(Pose_C(0.0,raw_data[0:3],raw_data[3:7]))
	
	abs_openv_test = OpenVSLAM_test_odom.getRelativeOdomInv()
	abs_openv_test.rotate(0.0,0.0,-0.5)
	abs_openv_test.plot(1)
#ORB SLAM(test)
	orb_test_odom = Odometry_C()
	orb_test_odom.name = 'ORB SLAM2(test)'
	for raw_data in x_test:
		orb_test_odom.array.append(Pose_C(0.0,raw_data[7:10],raw_data[10:]))
	
	abs_orb_test = orb_test_odom.getRelativeOdomInv()
	abs_orb_test.rotate(0.0,0.0,-0.35)
	abs_orb_test.plot(1)
	"""
#予測経路作成
#学習データ
			
	train_predict_odom = Odometry_C()
	train_predict_odom.name = 'prediciton(train)'	
	#train_predict_odom.array.append(Pose_C(0.0,np.zeros(3),np.array([0.0,0.0,0.0,1.0])))
	#train_predict_odom.array.append(OpenVSLAM_plot_odom.array[100])

	for i in range(y_train_pred.shape[0]):
		raw_data = y_train_pred[i]
		pose = Pose_C(0.0,raw_data[0:3],raw_data[3:])
		train_predict_odom.array.append(copy.deepcopy(pose))
	
	abs_train_predict_odom = train_predict_odom.getRelativeOdomInv()
	abs_train_predict_odom.rotate(0.0,0.0,-0.15)
	abs_train_predict_odom.plot(0)

#テストデータ
	test_predict_odom = Odometry_C()
	test_predict_odom.name = 'prediciton(test)'
	
	for i in range(y_test_pred.shape[0]):
		raw_data = y_test_pred[i]
		pose = Pose_C(0.0,raw_data[0:3],raw_data[3:])
		test_predict_odom.array.append(copy.deepcopy(pose))
	
	abs_test_predict_odom = test_predict_odom.getRelativeOdomInv()
	abs_test_predict_odom.rotate(0.0,0.0,-0.5)

	abs_test_predict_odom.plot(1)
#Plot setting

	#ax.scatter([sample[boundary][1][0]],[sample[boundary][1][1]],color='blue' ,zorder=100)
	#ax.legend(loc=2,fontsize = 7)
	#ax2.legend(loc=2,fontsize = 7)
	ax.set_xlabel('x[m]')
	ax.set_ylabel('y[m]')
	ax.set_xlim(-20.0,75.0)
	ax.set_ylim(-5.0,60.0)
	ax2.set_xlabel('x[m]')
	ax2.set_ylabel('y[m]')
	ax2.set_xlim(-20.0,75.0)
	ax2.set_ylim(-5.0,60.0)
	time = datetime.datetime.now()
	plt.tight_layout()
	plt.savefig(time.strftime ( '%Y–%m–%d-%H:%M:%S' ))
	
#fig = plt.figure(figsize=(5,5))
fig = plt.figure()
#plt.axes().set_aspect('equal')
#ax = fig.add_subplot(111, projection='3d')
ax = fig.add_subplot(211,aspect="equal")
ax2 = fig.add_subplot(212,aspect="equal")
if __name__ == "__main__":
	main()


