#!/usr/bin/env python

import os
import sys
import time
import copy
import datetime
import pandas as pd
import numpy as np
import tf
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


	
def readAccelFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.linear_acceleration.x':'x','field.linear_acceleration.y':'y','field.linear_acceleration.z':'z'})
	

	return df_new[['stamp','x','y','z']].values


def timestampFromStartTime(mat):
	div_num = 1000000000.0
	origin = mat[0][0] / div_num
	for i in range(mat.shape[0]):
		mat[i][0] = mat[i][0]/div_num -origin		

	return mat

def removeOffset(mat,offset_x,offset_y,offset_z):
	for i in range(mat.shape[0]):
		mat[i][1:4] = mat[i][1:4]-np.array([offset_x,offset_y,offset_z])
	return mat
	
		
def getOrientation(mat,window):
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
'''
def findPeakTime(win_array,acc_peak):

	max_idx = np.argmax(win_array[:,1]) #窓の中で最大の値をとるインデックス取得
	print(max_idx)
	if(win_array[max_idx,1] > acc_peak):
		return win_array[max_idx,0]
	else:
		return None
'''

def findPeakTime(acc_z,acc_peak,N):
	#find peak
	peak_idx_list = []
	for i in range(N//2,acc_z.shape[0]-N//2):
		offset = np.argmax(acc_z[i-N//2:i+N//2,1])
		max_idx = i-N//2 + offset
		if offset != N//2 or acc_z[max_idx,1] < acc_peak: #中央値でなない　or 閾値より小さい
			continue
		else:
			peak_idx_list.append(max_idx)
	return peak_idx_list
	
	

def findFootStep(acc_z,N):
	peak_time_list = []
	#find peak
	peak_idx_list = findPeakTime(acc_z,0.5,N)
	'''
	pre_peak_time = 0.0
	for i in range(N//2,acc_z.shape[0]-N//2):
		peak_time = findPeakTime(acc_z[i-N//2:i+N//2],0.5)
		if peak_time == None:
			continue
		elif i==N//2: #初回
			peak_idx = np.where(acc_z[:,0] == peak_time)[0][0]
			peak_time_list.append(peak_time)
			peak_idx_list.append(peak_idx)
			pre_peak_time = peak_time
		else:
			if peak_time != pre_peak_time: #前と違うピークを検出したら
				peak_idx = np.where(acc_z[:,0] == peak_time)[0][0]
				peak_time_list.append(peak_time)
				peak_idx_list.append(peak_idx)
				pre_peak_time = peak_time 
		'''

	return peak_idx_list
		
	

'''
	foot_step_idx_list = []
	for i in range(winsize//2,acc_z.shape[0]-winsize//2):
		win_array = acc_z[i:i+winsize,:]
		min_idx = np.argmin(win_array[:,1])
		if min_idx ==	winsize//2 and win_array[min_idx,1] < -1.0 :
			foot_step_idx_list.append(win_array[min_idx,0])
'''
	
	#return foot_step_idx_list


	
	
		
	
def main():
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
	assert len(args)>=2,'you must specify the arugument'
	#filename = os.path.normpath(os.path.join(os.getcwd(),args[1]))
	filename = sys.argv[1]
	print('loading' + filename)
	raw_data = readAccelFromCSV(filename)
	raw_data_timescale = timestampFromStartTime(raw_data)
	acc_calib = removeOffset(raw_data_timescale,0.0,0.0,0.0)

	#キャリブレーション後のデータをプロット
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,1],label='x')
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,2],label='y')
	raw_data_ax.plot(acc_calib[:,0],acc_calib[:,3],label='z')
	raw_data_ax.legend()

	#世界座標系の加速度を求める
	rpy = getOrientation(acc_calib,20)
	print('roll')
	print(math.degrees(rpy[0]))
	print('pitch')
	print(math.degrees(rpy[1]))

	rot = getRotationMatrix(rpy[0],rpy[1],0.0)
	acc_gcs = np.zeros_like(acc_calib)
	acc_gcs[:,0] = acc_calib[:,0] #タイムスタンプ代入
	for i in range(acc_calib.shape[0]):
		acc_gcs[i,1:4] = np.dot(rot,acc_calib[i,1:4].T)

	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,1],label='x')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,2],label='y')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,3],label='z')
	gcs_data_ax.legend()

	#重力成分を抽出
	acc_gravity = lowPassFilter(acc_gcs[:,3],0.9) #一次元配列

	gravity_data_ax.plot(acc_gcs[:,0],acc_gravity)

	#重力成分以外を抽出
	acc_hpf = acc_gcs[:,3] - acc_gravity[:]
	
	acc_step = np.zeros((acc_hpf.shape[0],2))
	win_size = 9
	moving_average_filter = np.ones(win_size) / win_size
	
	acc_step[:,1] = np.convolve(acc_hpf,moving_average_filter,mode='same')
	acc_step[:,0] = acc_gcs[:,0]

	acc_step_clip = acc_step[1000:1800,:] #Clip

	step_data_ax.plot(acc_step_clip[:,0],acc_step_clip[:,1])


	step_idx= findFootStep(acc_step_clip,50)
	print(step_idx)
	#step_data_ax.vlines(x=step_time, ymin=-3,ymax=3,zorder=100,color='red')
	step_data_ax.scatter(acc_step_clip[step_idx,0],acc_step_clip[step_idx,1],color='red')

	raw_data_fig.savefig('raw_data')
	gcs_data_fig.savefig('gcs_data')
	gravity_data_fig.savefig('gravity_data')
	step_data_fig.savefig('step_data')

	
	

	acc_raw_array = []
	time_array = []




if __name__ == "__main__":
	main()
