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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
	def setTime(self,time):
		self.__timestamp = time 
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
	def __init__(self,filename=None,label=None):
		self.array = []
		self.name = ''
		if filename is not None:
			self.load(filename)
		if label is not None:
			self.name = label
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
		
		print(self.name + ' max_delta:' + str(max(min_delta_list)))
		print(self.name + ' size:' + str(len(self.array)) + '->' + str(len(new_array)))
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

	def plot(self,ax):
		x_list = []
		y_list = [] 
		z_list = []
		for tmp in self.array:
			x_list.append(tmp.getPos()[0])
			y_list.append(tmp.getPos()[1])
			z_list.append(tmp.getPos()[2])

		ax.plot(x_list,y_list,label = self.name)
	
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
	
	def calMaxSpeed(self,window):
		max_velo = 0.0
		for i in range(100,len(self.array)):
			delta = self.array[i].getPos()-self.array[i-window].getPos()
			delta_time = abs(self.array[i].getTime()-self.array[i-window].getTime())
			velo = np.linalg.norm(delta)/delta_time
			if(max_velo < velo):
				max_velo = velo

		return max_velo
	
	def maxDeltaTime(self):
		max_delta = 0.0
		for i in range(1,len(self.array)):
			delta = self.array[i].getTime() - self.array[i-1].getTime()
			delta = abs(delta)
			if max_delta<delta:
				max_delta= delta
			return max_delta
	
	def meanDeltaTime(self):
		sum_time = 0.0
		cnt = 0
		for i in range(1,len(self.array)):
			delta = self.array[i].getTime() - self.array[i-1].getTime()
			delta = abs(delta)
			sum_time += delta
			cnt = cnt + 1
		return sum_time/cnt



#ある局所空間での類似性解析
#戻り値0->非類似　1->類似
def localClustering(odom1_rel,odom2_rel,begin,end,threshold):
	delta = []
	d_delta = []
	for i in range(begin,end):
		delta.append(odom1_rel.array[i].getPos() - odom2_rel.array[i].getPos())#位置関係取得
		
	for i in range(len(delta)-1):#フレーム間の位置関係の差
		d_delta = delta[i] - delta[i+1]
		if(np.linalg.norm(d_delta)>threshold):
			return 0
	
	return 1

def calCosSim(v1,v2):
	return np.dot(v1,v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def ClusteringFromTwoFrame(odom1_rel,odom2_rel,begin,threshold):
	delta1 = (odom1_rel.array[begin].getPos() - odom2_rel.array[begin].getPos())#位置関係取得
	delta2 = (odom1_rel.array[begin+1].getPos() - odom2_rel.array[begin+1].getPos())#位置関係取得
	sim = calCosSim(delta1,delta2)
	sim = abs(sim)
	print(sim)
	if(sim>threshold):
		return 1
	else:
		return 0
	'''	
	d_delta = delta1 - delta2
	if(np.linalg.norm(d_delta)>threshold):
		return 0
	else:
		return 1
	'''
#TODO
def calDiffVectorMap_v2(odom1,odom2,win_distance,d_n,N):
	kankei_vector = []
	range_plot = []
	flag = False
	for i in range(N):
		origin_pose1 = odom1.array[i]
		origin_pose2 = odom2.array[i]
		if flag==True:
			break
		for j in range(i,N,d_n):#d_nずつずらす
			distance = np.linalg.norm(odom1.array[j].getPos() - origin_pose1.getPos())
			if distance >= win_distance:
				target_pose1 = odom1.array[j]
				target_pose2 = odom2.array[j]
				break
			if j==N-1:
				flag = True
				

		line = np.concatenate([origin_pose1.getPos(),origin_pose2.getPos(),target_pose1.getPos(),target_pose2.getPos()],0)
		range_plot.append(line)
		relative_pose1 = target_pose1.getRelativePose(origin_pose1)
		relative_pose2 = target_pose2.getRelativePose(origin_pose2)	
		delta = relative_pose1.getPos() - relative_pose2.getPos()
		kankei_vector.append(delta)
	#分散
	array = np.array(kankei_vector)
	cov = np.cov(array[:,0],array[:,1])
	#平均
	mean = np.mean(array[:,0:2],axis=0)
	return kankei_vector,mean,cov,range_plot

#k:window size N:max size d_n kizami 
def calDiffVectorMap(odom1,odom2,k,d_n,N):
	kankei_vector = []
	cnt = 0
	for i in range(0,N-k,d_n):
		origin_pose1 = odom1.array[i]	
		target_pose1 = odom1.array[i+k]
		relative_pose1 = target_pose1.getRelativePose(origin_pose1)
		#ax3.scatter(relative_pose1.getPos()[0],relative_pose1.getPos()[1],s=1)

		origin_pose2 = odom2.array[i]
		target_pose2 = odom2.array[i+k]
		relative_pose2 = target_pose2.getRelativePose(origin_pose2) 
		#ax3.scatter(relative_pose2.getPos()[0],relative_pose2.getPos()[1],s=1)

		delta = relative_pose1.getPos() - relative_pose2.getPos()
		delta_normal = delta / (1/2*(np.linalg.norm(relative_pose1.getPos())+np.linalg.norm(relative_pose2.getPos())))

		kankei_vector.append(delta_normal)
		cnt = cnt + 1

	#分散
	array = np.array(kankei_vector)
	cov = np.cov(array[:,0],array[:,1])
	#平均
	mean = np.mean(array[:,0:2],axis=0)
	return kankei_vector,mean,cov

def transform(x_list,y_list,theta):
	new_xlist = []
	new_ylist = []

	for x,y in zip(x_list,y_list):
		new_xlist.append(x*math.cos(theta)-y*math.sin(theta))
		new_ylist.append(x*math.sin(theta)+y*math.cos(theta))
	
	return new_xlist,new_ylist

def calEllipse(mean,cov):
	val,vec = np.linalg.eigh(cov)	#固有値
	kai2 = 9.21034
	if val[0] > val[1]:
		a = math.sqrt(kai2*val[0])#長辺
		b = math.sqrt(kai2*val[1])#短辺
		large_vec = vec[0] 
	else :
		a = math.sqrt(kai2*val[1])#長辺
		b = math.sqrt(kai2*val[0])#短辺
		large_vec = vec[1]
		
	theta = np.linspace(0,2*np.pi,65)
	x = a * np.cos(theta)
	y = b * np.sin(theta)
	yaw = np.arctan2(large_vec[1],large_vec[0])
	new_x,new_y = transform(x,y,yaw)
	new_x = new_x + mean[0]
	new_y = new_y + mean[1]
	return new_x,new_y
	
	
def checkDeltaTime(odom1,odom2):
	min_size = min(odom1.size(),odom2.size())
	min_d_time = sys.float_info.max 
	for i in range(min_size):
		d_time = abs(odom1.array[i].getTime() - odom2.array[i].getTime())
		if(min_d_time > d_time):
			min_d_time = d_time
	
	return min_d_time

def plotKankeiVector(kankeivec,mean,cov,title,filename):
	figure = plt.figure(figsize = (5, 5))
	a = figure.add_subplot(111,aspect="equal")
	for i in range(len(kankeivec)):
		a.scatter(kankeivec[i][0],kankeivec[i][1],s=1)


	x,y = calEllipse(mean,cov)
	a.plot(x,y)
	limit = 5.0
	a.set_xlim(-limit,limit)
	a.set_ylim(-limit,limit)
	a.set_title(title)
	a.grid()

	figure.savefig(filename,dpi = 300)
def compareTwoOdom(odom1,odom2,base_filename):
	fig = plt.figure()
	min_time = max(odom1.array[0].getTime(),odom2.array[0].getTime())
	odom1.extract(min_time,0.2)
	odom2.extract(min_time,0.2)
	odom1 = odom1.split(0,1200)
	odom2 = odom2.split(0,1200)
	#kankei_vector,mean,cov = calDiffVectorMap(odom1,odom2,50,1,850)
	kankei_vector,mean,cov,range_plot = calDiffVectorMap_v2(odom1,odom2,10.0,1,1200)
	x,y = calEllipse(mean,cov)
	for i in range(len(kankei_vector)):
		fig.clf()
		ax_odom = fig.add_subplot(121,aspect='equal')
		ax_kankeivec = fig.add_subplot(122,aspect='equal')
		#ax_relpos = fig.add_subplot(224,aspect='equal')

		#Plot setting
		ax_odom.grid()	
		limit = 5.0
		ax_kankeivec.set_xlim(-limit,limit)
		ax_kankeivec.set_ylim(-limit,limit)
		ax_kankeivec.grid()
		#ax_relpos.grid()
		#ax_relpos.set_xlim(0,100)
		#ax_relpos.set_ylim(-100,100)


		odom1.plot(ax_odom)
		odom2.plot(ax_odom)
		#odom1_relpos = odom1.array[50+i].getRelativePose(odom1.array[i])
		#odom2_relpos = odom2.array[50+i].getRelativePose(odom2.array[i])
		
		#比較領域表示
		ax_odom.scatter(range_plot[i][0],range_plot[i][1],s=1,zorder=100,color='r')
		ax_odom.scatter(range_plot[i][3],range_plot[i][4],s=1,zorder=101,color='r')
		ax_odom.scatter(range_plot[i][6],range_plot[i][7],s=1,zorder=50,color='g')
		ax_odom.scatter(range_plot[i][9],range_plot[i][10],s=1,zorder=51,color='g')
		#相対移動表示
		#ax_relpos.scatter(odom1_relpos.getPos()[0],odom1_relpos.getPos()[1])
		#ax_relpos.scatter(odom2_relpos.getPos()[0],odom2_relpos.getPos()[1])
		#楕円
		ax_kankeivec.plot(x,y)
		#位置関係ベクトル表示
		for data in kankei_vector:
			ax_kankeivec.scatter(data[0],data[1],s=1,color='b')
		ax_kankeivec.scatter(kankei_vector[i][0],kankei_vector[i][1],s=2,color='r')

		index_str = '{0:04d}'.format(i)
		filename = base_filename + index_str 
		fig.savefig(filename)
		print('save:' + filename)

		
		
def analysis(odom_list):
	min_time = 0.0
	for odom in odom_list:
		time = odom.array[0].getTime()
		if min_time<time:
			min_time = time
	
	for odom in odom_list:
		odom.extract(min_time,0.2)
		odom = odom.split(0,850)	
		
	for odom1 in odom_list:
		for odom2 in odom_list:
			if odom1.name != odom2.name:
				kankei_vector,mean,cov,range_plot = calDiffVectorMap_v2(odom1,odom2,10.0,1,850)
				print('mean:' + str(mean))
				print('cov:' + str(cov))
				title = odom1.name + ' ' + odom2.name
				filename = odom1.name + '_' + odom2.name
				plotKankeiVector(kankei_vector,mean,cov, title,filename) 
		

def main():
	#オドメトリ表示用
	fig = plt.figure()
	ax = fig.add_subplot(111,aspect='equal')

	#lio_odom = Odometry_C(filename='lio_loop.csv', label='Lio')

	#wheel_odom = Odometry_C(filename='wheel_odom_loop.csv',label='Wheel')
	
	ndt_odom = Odometry_C(filename='~/csv/kgn_0223/ndt.csv',label='NDT mapping')

	openvslam_odom = Odometry_C(filename='~/csv/kgn_0223/openv_r1.csv',label='OpenVSLAM')
	scale = ndt_odom.length()/openvslam_odom.length()
	print('scale')
	print(scale)
	openvslam_odom.setScale(125)
	openvslam_odom.rotate(0.0,0.0,0.05)

	ndt_odom.plot(ax)
	openvslam_odom.plot(ax)
	ax.legend()

	#orb_zed_odom = Odometry_C(filename='./orb_zed2_ib_loop_localize_r05.csv',label='ORB SLAM(ZED)')
	#orb_zed_odom.setScale(lio_odom.length() / orb_zed_odom.length())
	#orb_zed_odom.rotate(0.0,0.0,0.05)
	#print('orb max dalta time' + str(orb_zed_odom.maxDeltaTime()))
	#print('orb mean dalta time ' + str(orb_zed_odom.meanDeltaTime()))

	
	

	odom_list = [ndt_odom,openvslam_odom]




	compareTwoOdom(ndt_odom,openvslam_odom,'ndt_openv')
	'''	
	min_time = max(openvslam_odom.array[0].getTime(),lio_odom.array[0].getTime(),orb_zed_odom.array[0].getTime())
	lio_odom.extract(min_time,0.2)
	lio_time_shift.extract(min_time,0.2)
	openvslam_odom.extract(min_time,0.2)
	orb_zed_odom.extract(min_time,0.2)
	wheel_odom.extract(min_time,0.2)

	new_open = openvslam_odom.split(0,850)
	new_lio = lio_odom.split(0,850)
	new_orb = orb_zed_odom.split(0,850)
	new_lio_shift = lio_time_shift.split(0,850)
	new_wheel = wheel_odom.split(0,850)
	new_lio.plot(ax)
	new_open.plot(ax)
	new_orb.plot(ax)
	new_wheel.plot(ax)
	'''

	#analysis(odom_list)	
	'''
	min_size = min(new_open.size(),new_lio.size())
	kankei_vectorAB,meanAB,covAB = calDiffVectorMap(new_lio,new_open,50,1,min_size)
	kankei_vectorAC,meanAC,covAC = calDiffVectorMap(new_lio,new_orb,50,1,min_size)
	kankei_vectorBC,meanBC,covBC = calDiffVectorMap(new_open,new_orb,50,1,min_size)
	plotKankeiVector(kankei_vectorAB,meanAB,covAB,'lio openv','./plot/LioOpenv.png')
	plotKankeiVector(kankei_vectorAC,meanAC,covAC,'lio orb','./plot/LioORB.png')
	plotKankeiVector(kankei_vectorBC,meanBC,covBC,'openvslam orb','./plot/OpenVSLAM.png')
	'''
	time = datetime.datetime.now()
#Plot setting
	ax.grid()
	fig.savefig('./' + time.strftime ( '%Y–%m–%d-%H:%M:%S' ))

	
if __name__ == "__main__":
	main()


