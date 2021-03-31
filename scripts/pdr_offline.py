#!/usr/bin/env python

import os
import sys
import time
import copy
import pandas as pd
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


	
def readAccelFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.linear_acceleration.x':'x','field.linear_acceleration.y':'y','field.linear_acceleration.z':'z'})
	

	return df_new[['stamp','x','y','z']].values

def readGNSSVeloFromCSV(filename):
	df = pd.read_csv(filename)
	df_new = df.rename(columns={'field.header.stamp':'stamp','field.accel.accel.linear.x':'x','field.accel.accel.linear.y':'y','field.accel.accel.linear.z':'z'})
	return df_new[['stamp','x','y','z']].values


def timestampFromStartTime(mat,origin_time):
	div_num = 1000000000.0
	for i in range(mat.shape[0]):
		mat[i][0] = (mat[i][0] - origin_time) / div_num

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
	peak_idx_list = findPeakTime(acc_z,0.5,N)
	p2p_idx_list = findPeak2PeakTime(acc_z,1.0,N)
	slope_idx_list = findSlopeTime(acc_z,N)

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
	
	return step_idx_list

#積分(台形法)
def integration(t_list,f_list):
	ret = 0.0
	for i in range(1,len(t_list)):
		dt = t_list[i] - t_list[i-1]
		area = (f_list[i-1] + f_list[i])*dt/2.0
		ret += area
	return ret


def calFootLength(acc_step,step_idx,K):
	ret = []
	acc_max = np.amax(acc_step[step_idx[0]:,1])
	acc_min = np.amin(acc_step[step_idx[0]:,1])
	ret.append(K*(acc_max-acc_min)**(1/4))
	for i in range(len(step_idx)-1):
		acc_max = np.amax(acc_step[step_idx[i]:step_idx[i+1],1])
		acc_min = np.amin(acc_step[step_idx[i]:step_idx[i+1],1])
		ret.append(K*(acc_max-acc_min)**(1/4))
	
	return ret

def calFootVelo(acc_step,step_idx,K):
	step_num = len(step_idx) #歩数
	ret = np.zeros((step_num,2))
	acc_max = np.amax(acc_step[step_idx[0]:,1])
	acc_min = np.amin(acc_step[step_idx[0]:,1])
	length = K*(acc_max-acc_min)**(1/4)
	for i in range(len(step_idx)-1):
		acc_max = np.amax(acc_step[step_idx[i]:step_idx[i+1],1])
		acc_min = np.amin(acc_step[step_idx[i]:step_idx[i+1],1])
		length = K*(acc_max-acc_min)**(1/4)
		velo = length / (acc_step[step_idx[i+1],0] - acc_step[step_idx[i],0])

		timestamp = (acc_step[step_idx[i+1],0] + acc_step[step_idx[i],0])/2
		ret[i,0] = timestamp
		ret[i,1] = velo

	return ret
	
def calPrameterK(acc_step,step_idx,velo_data):
	print()

def velo2Horizon(velo_mat): #速度を合成
	ret = np.zeros((velo_mat.shape[0],2))
	ret[:,0] = velo_mat[:,0] #copy timestamp
	for i in range(velo_mat.shape[0]):
		ret[i,1] = math.sqrt(velo_mat[i,1]**2+velo_mat[i,2]**2)
	
	return ret

def timeMatch(acc_step,velo_data)
	print()
	
def main():
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
	print(velo_horizon_data)




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

	if end_idx == None:
		acc_calib_sliced = acc_calib[start_idx:,:]
	else:
		acc_calib_sliced = acc_calib[start_idx:end_idx,:]

	acc_gcs = np.zeros_like(acc_calib_sliced)
	acc_gcs[:,0] = acc_calib_sliced[:,0] #タイムスタンプ代入
	for i in range(acc_calib_sliced.shape[0]):
		acc_gcs[i,1:4] = np.dot(rot,acc_calib_sliced[i,1:4].T)

	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,1],label='x')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,2],label='y')
	gcs_data_ax.plot(acc_gcs[:,0],acc_gcs[:,3],label='z')
	gcs_data_ax.legend()

	#重力成分を抽出
	acc_gravity = lowPassFilter(acc_gcs[:,3],0.9) #一次元配列

	gravity_data_ax.plot(acc_gcs[:,0],acc_gravity)

	#重力成分以外を抽出 acc_hpf:一次元配列
	#acc_hpf = acc_gcs[:,3] - acc_gravity[:]
	acc_hpf = acc_gravity - np.full_like(acc_gravity,9.8)
	
	acc_step = np.zeros((acc_hpf.shape[0],2))
	win_size = 9
	moving_average_filter = np.ones(win_size) / win_size
	
	acc_step[:,1] = np.convolve(acc_hpf,moving_average_filter,mode='same')
	acc_step[:,0] = acc_gcs[:,0]


	step_data_ax.plot(acc_step[:,0],acc_step[:,1])


	step_idx= findFootStep(acc_step,50)
	print('step index')
	print(step_idx)
	print('total step:',len(step_idx))
	#step_data_ax.vlines(x=step_time, ymin=-3,ymax=3,zorder=100,color='red')
	step_data_ax.scatter(acc_step[step_idx,0],acc_step[step_idx,1],color='red')
	#length = calFootLength(acc_step,step_idx,0.52)
	hoge = calFootVelo(acc_step,step_idx,0.52)
	print(hoge)

	raw_data_fig.savefig('raw_data')
	gcs_data_fig.savefig('gcs_data')
	gravity_data_fig.savefig('gravity_data')
	step_data_fig.savefig('step_data')

	
	





if __name__ == "__main__":
	main()
