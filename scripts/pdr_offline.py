#!/usr/bin/env python

import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import math
import sympy #sec
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

acc_offset = np.array([0.09,-0.07,0.07])



def kalmanFilter(imu_filename,mag_filename):
	fig_tmp = plt.figure()
	ax_tmp = fig_tmp.add_subplot(1,1,1)

	#Read CSV
	acc_data = readAccelFromCSV(imu_filename)
	gyro_data = readGyroFromCSV(imu_filename)
	mag_data = readMagFromCSV(mag_filename)

	#Init time
	start_time = acc_data[0,0]
	acc_data = timestampFromStartTime(acc_data,start_time)
	gyro_data = timestampFromStartTime(gyro_data,start_time)
	mag_data = timestampFromStartTime(mag_data,start_time)


	#観測したセンサデータの時系列を求める
	imu_len = acc_data.shape[0]
	mag_len = mag_data.shape[0]
	max_len = min(imu_len,mag_len)
	sensor_list = []
	pre_time = 0.0
	sensor_list.append([0.0,'imu',0])
	mag_idx = 0
	imu_idx = 0
	while mag_idx < max_len or imu_idx < max_len:
		mag_time = mag_data[mag_idx,0]
		imu_time = acc_data[imu_idx,0]
		if imu_time < mag_time: #imuが先なら
			pre_time = imu_time
			sensor_list.append([imu_time,'imu',imu_idx])
			imu_idx+=1
		else: #magが先なら
			pre_time = mag_time
			sensor_list.append([mag_time,'mag',mag_idx])
			mag_idx+=1

			

	initial_x = np.array([0,0,0])#roll pitch yaw
	x = initial_x.T	
	A = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
	P = np.array([[0.01,0.,0.],[0.,0.01,0.],[0.,0.,0.01]])
	Q = np.array([[0.001,0.,0.],[0.,0.001,0.],[0.,0.,0.0001]])
	R = np.array([[0.1,0.,0.],[0.,0.1,0.],[0.,0.,0.1]])

	'''
	x_tmp = np.zeros(3).T
	gyro_rpy = np.zeros((gyro_data.shape[0],4))
	
	for i in range(1,gyro_data.shape[0]):
		dt = gyro_data[i,0]-gyro_data[i-1,0]
		#B行列更新	
		B_tmp = np.array([[1,math.sin(x_tmp[0])*math.tan(x_tmp[1]),math.cos(x_tmp[0])*math.tan(x_tmp[1])],\
		[0,math.cos(x_tmp[0]),-math.sin(x_tmp[0])], \
		[0,math.sin(x_tmp[0])*sympy.sec(x_tmp[1]),math.cos(x_tmp[0])*sympy.sec(x_tmp[1])]])
		u = gyro_data[i-1,1:4].T
		x_n = np.dot(A,x_tmp)+np.dot(B_tmp,u*dt)
		x_tmp = x_n
		gyro_rpy[i,:] = np.array([gyro_data[i,0],x_n[0],x_n[1],x_n[2]])
	
	ax_tmp.plot(gyro_rpy[:,0],gyro_rpy[:,1],label='gyro_x')
	ax_tmp.plot(gyro_rpy[:,0],gyro_rpy[:,2],label='gyro_y')
	ax_tmp.plot(gyro_rpy[:,0],gyro_rpy[:,3],label='gyro_z')
	'''


	C_mag = np.array([[0,0,0],[0,0,0],[0,0,1]])
	C_imu = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]])



	'''
	test_rpy = np.zeros((acc_data.shape[0],3))
	for i in range(acc_data.shape[0]):
		test_rpy[i,:] = getOrientation(acc_data[i,1:4])
	ax_tmp.plot(acc_data[:,0],test_rpy[:,0],label='roll_acc')
	ax_tmp.plot(acc_data[:,0],test_rpy[:,1],label='pitch_acc')
	ax_tmp.plot(acc_data[:,0],test_rpy[:,2],label='yaw_acc')
	'''

#cal initial rpy
	rpy = getOrientation(acc_data[0,1:4])
	x[0] = rpy[0]
	x[1] = rpy[1]
	x_filtered = x

	tmp = np.array([acc_data[0,0],x[0],x[1],x[2]])
	ret = np.zeros((len(sensor_list),4))
	ret[0,:] = tmp

	imu_idx = 0
	mag_idx = 0
	pre_time = sensor_list[0][0]

	for i,sensor in enumerate(sensor_list,1):
		#B行列更新	
		if sensor[1] == 'imu' :
			B = np.array([[1,math.sin(x_filtered[0])*math.tan(x_filtered[1]),math.cos(x_filtered[0])*math.tan(x_filtered[1])],\
			[0,math.cos(x_filtered[0]),-math.sin(x_filtered[0])], \
			[0,math.sin(x_filtered[0])*sympy.sec(x_filtered[1]),math.cos(x_filtered[0])*sympy.sec(x_filtered[1])]])
			imu_idx = sensor[2]
		elif sensor[1] == 'mag' :
			B = np.zeros((3,3))
			mag_idx = sensor[2]
		else:
			print('sensor is not mag and imu')
			sys.exit()

		#deltaT計算
		delta = sensor[0] - pre_time

#Prediction
		u = gyro_data[imu_idx,1:4].T
		x_ = np.dot(A,x_filtered) + np.dot(B,u*delta)
		P = np.dot(np.dot(A,P),A.T)


#Filtering
			
#get Observation
		if sensor[1] == 'imu' :
			rpy = getOrientation(acc_data[imu_idx,1:4])
			y = rpy.T
			C = C_imu
			'''
			if np.linalg.norm(acc_data[imu_idx,1:4])>9.9 or np.linalg.norm(acc_data[imu_idx,1:4])<9.7:
				print('over')
				y = np.array([0,0,0]).T
				C = np.zeros((3,3))
			else:
				C = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]])
			'''
		else: #mag
			#get yaw
			R_mag2world = getRotationMatrix(x_filtered[0],x_filtered[1],0.0)
			mag_vector = np.dot(R_mag2world,mag_data[mag_idx,1:4].T)
			mag_yaw = math.atan(-mag_vector[1]/mag_vector[0])
			C = C_mag
			y = np.array([0.0,0.0,mag_yaw]).T

#calculate Kalman gain
		CPC = np.dot(C,np.dot(P,C.T))
		K = np.dot(np.dot(P,C.T),np.linalg.inv(CPC+R))
		x_filtered = x_ + np.dot(K,y-np.dot(C,x_))

		tmp = np.array([sensor[0],x_filtered[0],x_filtered[1],x_filtered[2]])
		ret[i-1,:] = tmp
		pre_time = sensor[0]
	
	ax_tmp.plot(ret[:,0],ret[:,1],label='roll')
	ax_tmp.plot(ret[:,0],ret[:,2],label='pitch')
	ax_tmp.plot(ret[:,0],ret[:,3],label='yaw')
	ax_tmp.set_ylim(-3.14,3.14)

	ax_tmp.legend()

	fig_tmp.savefig('filtered',dpi=100)

	


def readLidarPosFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.pose.pose.position.x':'x','field.pose.pose.position.y':'y','field.pose.pose.position.z':'z'})
	mat = df_new[['stamp','x','y','z']].values
	return mat

def calVeloFromPos(mat):
	average = 1
	velo_mat = np.zeros((mat.shape[0]-average,2))
#速度計算
	for i in range(average,mat.shape[0]):
		delta_t = mat[i,0]-mat[i-average,0]
		delta_pos = mat[i,1:4]-mat[i-average,1:4]
		delta_pos_norm = math.sqrt(delta_pos[0]**2+delta_pos[1]**2+delta_pos[2]**2)
		#delta_pos_norm = math.sqrt(delta_pos[0]**2+delta_pos[1]**2)
		velo = delta_pos_norm/delta_t
		velo_mat[i-average,:] = np.array([mat[i,0],velo])
	return velo_mat
		

def readGyroFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.angular_velocity.x':'x','field.angular_velocity.y':'y','field.angular_velocity.z':'z'})
	return df_new[['stamp','x','y','z']].values
def readMagFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.magnetic_field.x':'x','field.magnetic_field.y':'y','field.magnetic_field.z':'z'})
	return df_new[['stamp','x','y','z']].values
	
def readAccelFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.linear_acceleration.x':'x','field.linear_acceleration.y':'y','field.linear_acceleration.z':'z'})
	

	return df_new[['stamp','x','y','z']].values

def readGNSSVeloFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.accel.accel.linear.x':'x','field.accel.accel.linear.y':'y','field.accel.accel.linear.z':'z'})
	
	mat = df_new[['stamp','x','y','z']].values
	
	#同じ値の部分を排除
	delete_list = []
	pre_vec = mat[0,1:4]
	for i in range(1,mat.shape[0]):
		if(np.all(pre_vec == mat[i,1:4])):
			delete_list.append(i)
		pre_vec = mat[i,1:4]

	ret_mat = np.delete(mat,delete_list,axis = 0)
	print(ret_mat)
	return ret_mat


def timestampFromStartTime(mat,origin_time):
	div_num = 1000000000.0
	for i in range(mat.shape[0]):
		mat[i][0] = (mat[i][0] - origin_time) / div_num

	return mat

def removeOffset(mat,offset_x,offset_y,offset_z):
	for i in range(mat.shape[0]):
		mat[i][1:4] = mat[i][1:4]-np.array([offset_x,offset_y,offset_z])
	return mat
	
		
def getOrientation(acc):
	rpy = np.zeros(3)
	rpy[0] = math.atan2(acc[1],acc[2]) #tan(ay/az)
	rpy[1] = math.atan(-acc[0]/math.sqrt(acc[1]**2+acc[2]**2))
	rpy[2] = 0.0
	return rpy

	
def getAverageOrientation(mat,window):
	sum_var = np.zeros(3)
	rpy = np.zeros(3)
	
	for i in range(window):
		sum_var += mat[i,1:4]
	average_acc = sum_var/window
	rpy[0] = math.atan2(average_acc[1],average_acc[2]) #tan(ay/az)
	rpy[1] = math.atan(-average_acc[0]/math.sqrt(average_acc[1]**2+average_acc[2]**2))
	rpy[2] = 0.0
	return rpy	




def getRotationMatrix(roll,pitch,yaw):
	mat = np.zeros((3,3))
	rot_x = np.array([[1,0,0],[0,math.cos(roll),-math.sin(roll)],[0,math.sin(roll),math.cos(roll)]])
	rot_y = np.array([[math.cos(pitch),0,math.sin(pitch)],[0,1,0],[-math.sin(pitch),0,math.cos(pitch)]])
	rot_z = np.array([[math.cos(yaw),-math.sin(yaw),0],[math.sin(yaw),math.cos(yaw),0],[0,0,1]])
	return np.dot(rot_z,np.dot(rot_y,rot_x))


def lowPassFilter(mat,alpha): #一次元配列
	init_val = mat[0]
	ret = np.zeros_like(mat)
	ret[0] = init_val
	for i in range(1,mat.shape[0]):
		ret[i] = ret[i-1]*alpha + (1-alpha)*mat[i]

	return ret

def findPeakTime(acc_z,acc_peak,N):
	#find peak
	peak_idx_list = []
	for i in range(N//2,acc_z.shape[0]-N//2):
		offset = np.argmax(acc_z[i-N//2:i+N//2,1])
		max_idx = i-N//2 + offset
		if offset != N//2 or acc_z[max_idx,1] > acc_peak: #中央値でなない　or 閾値より小さい
			continue
		else:
			peak_idx_list.append(max_idx)
	return peak_idx_list
	
def findNegativePeakTime(acc_z,acc_peak,N):
	#find peak
	peak_idx_list = []
	for i in range(N//2,acc_z.shape[0]-N//2):
		offset = np.argmin(acc_z[i-N//2:i+N//2,1])
		min_idx = i-N//2 + offset
		if offset != N//2 or acc_z[min_idx,1] > acc_peak: #中央値でなない　or 閾値より小さい
			continue
		else:
			peak_idx_list.append(min_idx)
	return peak_idx_list
def findNegativeSlopeTime(acc_z,N):
	slope_idx_list = []
	for t in range(N//2,acc_z.shape[0]-N//2):
		sum_var1 = 0.0
		sum_var2 = 0.0
		for i in range(t-N//2,t-1):
			sum_var1 += acc_z[i+1,1] - acc_z[i,1]
		for i in range(t+1,t+N//2):
			sum_var2 += acc_z[i,1] - acc_z[i-1,1]
		if sum_var1 < 0 and sum_var2 > 0:
			slope_idx_list.append(t)
	
	return slope_idx_list
def findNegativePeak2PeakTime(acc_z,acc_pp,N):
	p2p_idx_list = []
	for t in range(N//2,acc_z.shape[0]-N//2):
		delta1 = []
		delta2 = []
		for i in range(1,N//2):
			delta1.append(acc_z[t,1] - acc_z[t+i,1])
			delta2.append(acc_z[t,1] - acc_z[t-i,1])
		max_delta1 = min(delta1)
		max_delta2 = min(delta2)
		if max_delta1 < acc_pp and max_delta2 < acc_pp:
			p2p_idx_list.append(t)

	return p2p_idx_list
def findPeak2PeakTime(acc_z,acc_pp,N):
	p2p_idx_list = []
	for t in range(N//2,acc_z.shape[0]-N//2):
		delta1 = []
		delta2 = []
		for i in range(1,N//2):
			delta1.append(acc_z[t,1] - acc_z[t+i,1])
			delta2.append(acc_z[t,1] - acc_z[t-i,1])
		max_delta1 = max(delta1)
		max_delta2 = max(delta2)
		if max_delta1 > acc_pp and max_delta2 > acc_pp:
			p2p_idx_list.append(t)

	return p2p_idx_list

def findSlopeTime(acc_z,N):
	slope_idx_list = []
	for t in range(N//2,acc_z.shape[0]-N//2):
		sum_var1 = 0.0
		sum_var2 = 0.0
		for i in range(t-N//2,t-1):
			sum_var1 += acc_z[i+1,1] - acc_z[i,1]
		for i in range(t+1,t+N//2):
			sum_var2 += acc_z[i,1] - acc_z[i-1,1]
		if sum_var1 > 0 and sum_var2 < 0:
			slope_idx_list.append(t)
	
	return slope_idx_list

def findFootStep(acc_z,N):
	#find peak
	#peak_idx_list = findPeakTime(acc_z,0.5,N)
	peak_idx_list = findNegativePeakTime(acc_z,-0.5,N)
	p2p_idx_list = findNegativePeak2PeakTime(acc_z,-0.5,N)
	slope_idx_list = findNegativeSlopeTime(acc_z,10)

	#Plot Peak
	peak_fig = plt.figure()
	peak_ax = peak_fig.add_subplot(1,1,1)
	peak_ax.plot(acc_z[:,0],acc_z[:,1])
	peak_ax.scatter(acc_z[peak_idx_list,0],acc_z[peak_idx_list,1],color='red')
	peak_fig.savefig('peak')
	#Plot Peak2Peak
	p2p_fig = plt.figure()
	p2p_ax = p2p_fig.add_subplot(1,1,1)
	p2p_ax.plot(acc_z[:,0],acc_z[:,1])
	p2p_ax.scatter(acc_z[p2p_idx_list,0],acc_z[p2p_idx_list,1],color='red')
	p2p_fig.savefig('peak2peak')

	#Plot Slope 
	slope_fig = plt.figure()
	slope_ax = slope_fig.add_subplot(1,1,1)
	slope_ax.plot(acc_z[:,0],acc_z[:,1])
	slope_ax.scatter(acc_z[slope_idx_list,0],acc_z[slope_idx_list,1],color='red')
	slope_fig.savefig('slope')
	#FUSION
	peak_idx_set = set(peak_idx_list)
	p2p_idx_set = set(p2p_idx_list)
	slope_idx_set = set(slope_idx_list)
	fusion_idx_set = peak_idx_set & p2p_idx_set & slope_idx_set
	step_idx_list = list(fusion_idx_set)

	step_idx_list.sort()


	#Plot step 
	step_fig = plt.figure()
	step_ax = step_fig.add_subplot(1,1,1)
	step_ax.plot(acc_z[:,0],acc_z[:,1])
	step_ax.scatter(acc_z[step_idx_list,0],acc_z[step_idx_list,1],color='red')
	step_fig.savefig('step')
	
	return step_idx_list

#積分(台形法)
def integration(t_list,f_list):
	ret = 0.0
	for i in range(1,len(t_list)):
		dt = t_list[i] - t_list[i-1]
		area = (f_list[i-1] + f_list[i])*dt/2.0
		ret += area
	return ret


def calFootLength(acc_step,step_idx,K,bias):
	ret = []
#初歩の処理
	acc_max = np.amax(acc_step[0:step_idx[0]+1,1])
	acc_min = acc_step[step_idx[0],1]
	ret.append(K*(acc_max-acc_min)**(1/4))
#初歩以外の処理
	for i in range(1,len(step_idx)):
		acc_max = np.amax(acc_step[step_idx[i-1]:step_idx[i]+1,1])
		acc_min = acc_step[step_idx[i],1]
		ret.append(K*(acc_max-acc_min)**(1/4)+bias)


	return ret



def calFootVelo(acc_step,step_idx,K,bias):
	step_num = len(step_idx) #歩数
	ret = np.zeros((step_num,2))
#初回なし
	'''
	acc_max = np.amax(acc_step[step_idx[0]:,1])
	acc_min = np.amin(acc_step[step_idx[0]:,1])
	length = K*(acc_max-acc_min)**(1/4)
	'''
	for i in range(1,len(step_idx)):
		acc_max = np.amax(acc_step[step_idx[i-1]:step_idx[i]+1,1])
		acc_min = acc_step[step_idx[i],1]
		length = K*(acc_max-acc_min)**(1/4)
		velo = length / (acc_step[step_idx[i],0] - acc_step[step_idx[i-1],0])

		timestamp = (acc_step[step_idx[i],0] + acc_step[step_idx[i-1],0])/2
		ret[i,0] = timestamp
		ret[i,1] = velo

	return ret

def calFootLength_TianModel(acc, step_idx,height,k):
	ret = []
	for i in range(1,len(step_idx)):
		delta =	acc[step_idx[i],0] - acc[step_idx[i-1],0] #タイムスタンプの差分を取得 
		freq = 1/delta
		length = k*height*math.sqrt(freq)
		ret.append(length)

	return ret



def leastSquaresMethod(x_list,y_list):
	x_cov = np.var(x_list)
	xy_cov = np.cov(np.array([x_list,y_list]))[0,0]
	print(xy_cov)
	x_mean = np.mean(x_list)
	y_mean = np.mean(y_list)
	
	slope = xy_cov/x_cov
	bias = y_mean-slope*x_mean

	return slope,bias

def leastSquaresMethodNobias(x_list,y_list):
	xy = 0.0
	x_2 = 0.0
	for x,y in zip(x_list,y_list):
		xy += x*y
		x_2 += x**2
	
	return xy/x_2


def matchTime(x,y,max_delay):
	ret = []
	for i in range(x.shape[0]):
		target_time = x[i,0]
		min_time = sys.float_info.max
		for j in range(y.shape[0]):
			delta_time = abs(y[j,0] - target_time)
			if min_time >= delta_time:
				min_time = delta_time
				min_idx = j
			else:
				if min_time<max_delay:
					ret.append((i,min_idx))
				break
	return ret





def calPrameterK(acc_step,step_idx,velo_data):
	step_num = len(step_idx) #歩数
	x_list = np.zeros((step_num,2))
	for i in range(len(step_idx)-1):
		acc_max = np.amax(acc_step[step_idx[i]:step_idx[i+1],1])
		acc_min = np.amin(acc_step[step_idx[i]:step_idx[i+1],1])
		tmp = (acc_max-acc_min)**(1/4)
		x = tmp / (acc_step[step_idx[i+1],0] - acc_step[step_idx[i],0])

		timestamp = (acc_step[step_idx[i+1],0] + acc_step[step_idx[i],0])/2
		x_list[i,0] = timestamp
		x_list[i,1] = x

	
	print('x_list')
	print(x_list)
	print('velo data')
	print(velo_data)

	idx = matchTime(x_list,velo_data,0.1)
	print(idx)
	x_plot = []
	y_plot = []
	for i in range(len(idx)):
		print(x_list[idx[i][0]])
		print(velo_data[idx[i][1]])
		x_plot.append(x_list[idx[i][0],1])
		y_plot.append(velo_data[idx[i][1],1])
	
	

	slope = leastSquaresMethodNobias(x_plot,y_plot)
	print('parameterK')
	print(slope)

	line_x = np.linspace(0,4,100)
	line_y = np.zeros_like(line_x)
	for i in range(line_x.shape[0]):
		line_y[i] = slope*line_x[i]

	
	
	fig_test = plt.figure()

	fig_ax = fig_test.add_subplot(1,1,1)
	fig_ax.scatter(x_plot,y_plot)
	fig_ax.plot(line_x,line_y)
	fig_ax.set_xlim(0,4)
	fig_ax.set_ylim(0,1.5)
	fig_ax.set_aspect('equal')
	fig_ax.grid()
	fig_test.savefig('plot')
	
	length = calFootLength(acc_step,step_idx,0.48,0)
	print('foot length')
	print(length)

def velo2Horizon(velo_mat): #速度を合成
	ret = np.zeros((velo_mat.shape[0],2))
	ret[:,0] = velo_mat[:,0] #copy timestamp
	for i in range(velo_mat.shape[0]):
		ret[i,1] = math.sqrt(velo_mat[i,1]**2+velo_mat[i,2]**2)
	
	return ret

def calAccStep(acc_calib,start_idx=0,end_idx=None):
	#世界座標系の加速度を求める
	rpy = getAverageOrientation(acc_calib,20)
	print('roll')
	print(math.degrees(rpy[0]))
	print('pitch')
	print(math.degrees(rpy[1]))

	rot = getRotationMatrix(rpy[0],rpy[1],0.0)

	if end_idx == None:
		acc_calib_sliced = acc_calib[start_idx:,:]
	else:
		acc_calib_sliced = acc_calib[start_idx:end_idx,:]

	acc_gcs = np.zeros_like(acc_calib_sliced)
	acc_gcs[:,0] = acc_calib_sliced[:,0] #タイムスタンプ代入
	for i in range(acc_calib_sliced.shape[0]):
		acc_gcs[i,1:4] = np.dot(rot,acc_calib_sliced[i,1:4].T)

	'''
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,1],label='x')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,2],label='y')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,3],label='z')
	gcs_data_ax.legend()
	'''

	#重力成分を抽出
	acc_gravity = lowPassFilter(acc_gcs[:,3],0.9) #一次元配列

	#gravity_data_ax.plot(acc_gcs[:,0],acc_gravity)

	#重力成分以外を抽出 acc_hpf:一次元配列
	acc_hpf = acc_gcs[:,3] - acc_gravity[:]
	#acc_hpf = acc_gravity - np.full_like(acc_gravity,9.8) #定数重力引くver
	
	acc_step = np.zeros((acc_hpf.shape[0],2))
	win_size = 9
	moving_average_filter = np.ones(win_size) / win_size
	
	acc_step[:,1] = np.convolve(acc_hpf,moving_average_filter,mode='same')
	acc_step[:,0] = acc_gcs[:,0]

	return acc_step

def StepCount(filename): #ステップ数を検証する為だけのテスト用関数
	raw_imu_data = readAccelFromCSV(filename)
	start_time = raw_imu_data[0,0]
	raw_imu_data_timescale = timestampFromStartTime(raw_imu_data,start_time)
	acc_calib = removeOffset(raw_imu_data_timescale,acc_offset[0],acc_offset[1],acc_offset[2])
	acc_step = calAccStep(acc_calib)
	step_idx= findFootStep(acc_step,50)
	step_cnt = len(step_idx)
	print(filename,':',step_cnt,'steps')

def addPlotSample(filename,x_plot,y_plot):
	
#データ下処理
	raw_imu_data = readAccelFromCSV(filename+'_imu.csv')
	start_time = raw_imu_data[0,0]
	raw_imu_data_timescale = timestampFromStartTime(raw_imu_data,start_time)
	acc_calib = removeOffset(raw_imu_data_timescale,acc_offset[0],acc_offset[1],acc_offset[2])
	acc_step = calAccStep(acc_calib)

	raw_velo_data = readGNSSVeloFromCSV(filename+'_velo.csv')
	velo_data = timestampFromStartTime(raw_velo_data,start_time)
	velo_horizon_data = velo2Horizon(velo_data)
	np.savetxt(filename+'gnss_velo.csv',velo_horizon_data,delimiter=',')
	'''
	raw_truth_pos_data = readLidarPosFromCSV(filename+'_slam.csv')
	truth_pos_data = timestampFromStartTime(raw_truth_pos_data,start_time)
	truth_velo_data = calVeloFromPos(truth_pos_data)
	'''
	
	#np.savetxt(filename+'slam_velo.csv',truth_velo_data,delimiter=',')

	
#ステップカウント
	step_idx= findFootStep(acc_step,50)

	step_num = len(step_idx) #歩数
	x_list = np.zeros((step_num-1,2))
#初回ステップは無視
	for i in range(1,len(step_idx)):
		acc_max = np.amax(acc_step[step_idx[i-1]:step_idx[i],1])
		acc_min = np.amin(acc_step[step_idx[i],1])
		tmp = (acc_max-acc_min)**(1/4)
		x = tmp / (acc_step[step_idx[i],0] - acc_step[step_idx[i-1],0])

		timestamp = (acc_step[step_idx[i],0] + acc_step[step_idx[i-1],0])/2
		x_list[i-1,0] = timestamp
		x_list[i-1,1] = x


	idx = matchTime(x_list,velo_horizon_data,0.1)
	#idx = matchTime(x_list,truth_velo_data,0.1)

	for i in range(len(idx)):
		x_plot.append(x_list[idx[i][0],1])
		y_plot.append(velo_horizon_data[idx[i][1],1])
		#y_plot.append(truth_velo_data[idx[i][1],1])
	
def calWalkingDistance(filename,K):
	raw_imu_data = readAccelFromCSV(filename)
	start_time = raw_imu_data[0,0]
	raw_imu_data_timescale = timestampFromStartTime(raw_imu_data,start_time)
	acc_calib = removeOffset(raw_imu_data_timescale,0.0,0.0,0.0)
	acc_step = calAccStep(acc_calib)
	step_idx= findFootStep(acc_step,50)
	print(len(step_idx))

	length = calFootLength(acc_step,step_idx,K,0)
	print(sum(length))	

def calDistanceFromSLAM(filename):
	mat = readLidarPosFromCSV(filename)
	length = 0.0
	for i in range(1,mat.shape[0]):
		pre_pos = mat[i-1,1:4]
		pos = mat[i,1:4]
		delta = pos - pre_pos
		length += math.sqrt(delta[0]**2+delta[1]**2+delta[2]**2)

	return length




	
def main():

	#calWalkingDistance(sys.argv[1],0.44)
	#return

	#kalmanFilter(sys.argv[1],sys.argv[2])

	sample = 200
	x_plot = []
	y_plot = []

	pallet = ['b','g','r','c','m','y','k']

	plot_fig = plt.figure()
	plot_ax = plot_fig.add_subplot(1,1,1)

	x_plot = []
	y_plot = []
	for i in range(1,len(sys.argv)):
		print('filename:',sys.argv[i])
		StepCount(sys.argv[i]+'_imu.csv')
		x_tmp = []
		y_tmp = []
		addPlotSample(sys.argv[i],x_tmp,y_tmp)
		plot_ax.scatter(x_tmp,y_tmp,s=2,color=pallet[i],label=sys.argv[i])
		x_plot.extend(x_tmp)
		y_plot.extend(y_tmp)
	#addPlotSample(sys.argv[2],x_plot,y_plot)

	slope = leastSquaresMethodNobias(x_plot,y_plot)
	print('parameterK')
	print(slope)

	
	line_x = np.linspace(0,4,100)
	line_y = np.zeros_like(line_x)
	for i in range(line_x.shape[0]):
		line_y[i] = slope*line_x[i]

	#残差計算
	'''
	error_sum = 0.0
	for i in range(len(x_plot)):
		ref_y = slope*x_plot[i]
		error_sum += y_plot[i] - ref_y
	
	error_ave = error_sum/len(x_plot)
	print('average error',error_ave)
	'''

	plot_ax.plot(line_x,line_y)
	#plot_ax.set_xlim(0,4)
	#plot_ax.set_ylim(0,2)
	plot_ax.set_aspect('equal')
	plot_ax.grid()
	plot_ax.legend()
	plot_ax.set_xlim(0,4.0)
	plot_ax.set_ylim(0,2.0)
	plot_fig.savefig('plot',dpi=300)




	
	'''
	np.set_printoptions(precision=3)
	np.set_printoptions(suppress=True)
	#Matplotlib
	raw_data_fig = plt.figure()
	raw_data_ax = raw_data_fig.add_subplot(1,1,1)

	gcs_data_fig = plt.figure()
	gcs_data_ax = gcs_data_fig.add_subplot(1,1,1)

	gravity_data_fig = plt.figure()
	gravity_data_ax = gravity_data_fig.add_subplot(1,1,1)

	step_data_fig = plt.figure()
	step_data_ax = step_data_fig.add_subplot(1,1,1)

	args = sys.argv

	if len(args)<=2 :
		print(args)
		print('you must specify the arugument')
		print('ex: filename start_idx end_idx')
		return
	filename_imu = sys.argv[1]
	filename_velo = sys.argv[2]
	if len(args)>=4:
		start_idx = int(sys.argv[3])
	else:
		start_idx = 0
	if len(args)>=4:
		end_idx = int(sys.argv[4])
	else:
		end_idx = None
	print('loading' + filename_imu)
	


	raw_imu_data = readAccelFromCSV(filename_imu)
	start_time = raw_imu_data[0,0]
	raw_imu_data_timescale = timestampFromStartTime(raw_imu_data,start_time)
	acc_calib = removeOffset(raw_imu_data_timescale,0.0,0.0,0.0)

	raw_velo_data = readGNSSVeloFromCSV(filename_velo)
	velo_data = timestampFromStartTime(raw_velo_data,start_time)
	velo_horizon_data = velo2Horizon(velo_data)




	#キャリブレーション後のデータをプロット
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,1],label='x')
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,2],label='y')
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,3],label='z')
	raw_data_ax.legend()


	acc_step = calAccStep(acc_calib)


	step_data_ax.plot(acc_step[:,0],acc_step[:,1])


	step_idx= findFootStep(acc_step,50)
	print('step index')
	print(step_idx)
	print('total step:',len(step_idx))
	#step_data_ax.vlines(x=step_time, ymin=-3,ymax=3,zorder=100,color='red')
	step_data_ax.scatter(acc_step[step_idx,0],acc_step[step_idx,1],color='red')
	#length = calFootLength(acc_step,step_idx,0.48,0)
	#print(sum(length))
	calPrameterK(acc_step,step_idx,velo_horizon_data)
	

	raw_data_fig.savefig('raw_data')
	gcs_data_fig.savefig('gcs_data')
	gravity_data_fig.savefig('gravity_data')
	step_data_fig.savefig('step_data')
	'''

	
	





if __name__ == "__main__":
	main()
