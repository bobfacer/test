# """
# Created on Thu Jun 20 14:38:14 2019
# @author: lukesmith
# """
# from random import uniform, shuffle
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns


# L = 100 # number of cells on road
# n_iters = 100 # no. iterations

# density = 0.2 # percentage of cars

# vmax = 5 #maximum velocity

# p = 0.3 #probability that a car will slow down


# car_num = int(density * L)
# ini = [0] * car_num + [-1] * (L - car_num) 
#     # creates an array with cars and empty spaces
# shuffle(ini) 
#     # This randomises the array to the sars are spread along the road.

# iterate = [ini]

# for i in range(n_iters):
#     prev,cur = iterate[-1],[-1] * L

#     for k in range(L):
#         if prev[k] > -1:
#             vi = prev[k]
#             d = 1
#             while prev[(k + d) % L] < 0:
#                 d += 1

#             vtemp = min(vi+1, d - 1, vmax)  #increase speed up to max. 
#                                             #cars do not move further than next car
#             v = max(vtemp - 1, 0) if uniform(0,1) < p else vtemp #probability p a car hits the brakes otherwise velocity is sustained
# #            if (k+v) < L:
# #                    cur[(k + v)] = v
#             if (k+v) < L:
#                 cur[(k + v)] = v # allows the cars to exit the screen
#             else:
#                 cur[(k)] = -1
      

#     iterate.append(cur)


# a = np.zeros(shape=(n_iters,L))
# for i in range(L):
#     for j in range(n_iters):
#         a[j,i] = 1 if iterate[j][i] > -1 else 0
 
# #plotting
# test = np.array(iterate)

# fig, ak = plt.subplots()
# sns.heatmap(data=test, vmin = 0,ax = ak,vmax = 19, cmap='rainbow') 
# #im = ak.imshow(test, cmap="gist_heat", interpolation="nearest")
# plt.xlabel("Space")
# plt.ylabel("Time")
# plt.show()
# print('finish')

"""
Created on Tue Apr 16 10:44:47 2019
@author: dddd
"""

#NaSch模型
#Author Dong Jiakuan
#Date 2019.04.20

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

############################ 函数定义 #########################################
# 根据只包含车路位置和速度的一维数组转换成包含速度和位置的二维数组
def transfer(vehicle_number,road_length,vehicle_position,vehicle_velocity):
    cells_data = np.zeros((2,road_length),dtype = int)
    for i in range(vehicle_number):
        index = int(vehicle_position[i])
        cells_data[0,index] = 1
        cells_data[1,index] = vehicle_velocity[i]
    return cells_data
    
############################ 参数定义 #########################################
v_max = 5 #最大速度
road_length = 2000 #道路长度
vehicle_length = 1 #车辆长度
p = 0.3 #随机慢化概率
#density = 0.3 #密度,后期可以改成一个列表，循环仿真
steps = 1000 #仿真步长
#vehicle_number = int(road_length*density) #车辆数
velocities = np.zeros(0,dtype=float)
# densities = np.linspace(0.02,1,num=49,endpoint=False)
# np.delete(densities,0)
densities = [0.09] #最大时候的density

truck_in = 120
truck_out = 420
truck_in_position = 1200
truck_out_position = 1500
last_position = truck_in_position
truck_v = 1
flag = 0

for density in densities:
############################ 声明过程数据 #####################################
    velocity_avg = 0
    vehicle_number = int(road_length*density)
    vehicle_velocity = np.zeros((vehicle_number),dtype = int) #存储车辆速度
    vehicle_position = np.zeros((vehicle_number),dtype = int) #存储车辆的位置
    data = np.zeros((0,2,road_length),dtype = int) #用于存储元胞过程状态数据

######################### 初始化 ##############################################
#速度是确定的，车辆位置也应当是确定的
#随机生成车辆位置，并记录到vehicle_position中，按照大小排序
    vehicle_position = np.array(random.sample(range(road_length),vehicle_number))
    vehicle_position.sort()
#赋予车辆随机初始速度
    vehicle_velocity = np.random.randint(v_max+1,size=vehicle_number)
#将初始化数据放入data中
    cells_data = transfer(vehicle_number,road_length,vehicle_position,vehicle_velocity)
    data = np.append(data,[cells_data],axis = 0)

    falg = 0
    log = np.zeros((road_length),dtype = int)
########################## 迭代 ###############################################
    for i in range(steps):
        if i%5==0 and vehicle_position[0]!=0:
            vehicle_position = np.append(vehicle_position,0)
            vehicle_position.sort()
            vehicle_velocity = np.insert(vehicle_velocity,0,1)
            vehicle_number = vehicle_number + 1
    #第一步：加速
        vehicle_velocity = np.min(np.vstack((vehicle_velocity+1,\
                                         v_max*np.ones_like(vehicle_velocity))),axis=0)
    #第二步：减速
        #二分把truck_in_position找到插入位置
        #truck_in_position插入
        #每次truck_in_position自增1
        #到达指定时间后移除truck_in_position
        if i==truck_in:
            vehicle_position = np.append(vehicle_position,truck_in_position)
            vehicle_position.sort()
            idx = np.argwhere(vehicle_position == truck_in_position)
            vehicle_velocity =np.insert(vehicle_velocity,idx[0][0],-10000)
            flag = 1
        elif i==truck_out:
            pos_idx = np.argwhere(vehicle_position==last_position)
            vehicle_position =np.delete(vehicle_position,pos_idx[0][0],0)
            idx_remove = np.argwhere(vehicle_velocity<0)
            vehicle_velocity =np.delete(vehicle_velocity,idx_remove[0][0],0) #执行删除 只删除了第2个数据
            flag = 0
        elif flag == 1:
            pos_idx = np.argwhere(vehicle_position==last_position)
            vehicle_position =np.delete(vehicle_position,pos_idx[0][0],0)
            vehicle_velocity =np.delete(vehicle_velocity,pos_idx[0][0],0)
            last_position = last_position + 1
            vehicle_position = np.append(vehicle_position,last_position)
            vehicle_position.sort()
            idx = np.argwhere(vehicle_position == last_position)
            vehicle_velocity =np.insert(vehicle_velocity,idx[0][0],-10000)
            vehicle_position.sort()


        dis = np.delete(vehicle_position,0)
        dis = np.append(dis,vehicle_position[0]+road_length) #如果不是为了计算环形，最后一个的dis可以无限大
        dis = dis - vehicle_position
        dis = dis - vehicle_length
        vehicle_velocity = np.min(np.vstack((vehicle_velocity,dis)),axis=0)
    # 第三步：随机慢化
        if flag==1:
            prob = np.random.random(vehicle_number+1)
            slow_c = np.zeros(vehicle_number+1)
        else:
            prob = np.random.random(vehicle_number)
            slow_c = np.zeros(vehicle_number)
        v_above = np.multiply(vehicle_velocity,np.array(prob>=p,dtype=int))
        if flag==1:
            v_above[idx[0][0]] = -10000
        v_below = np.multiply(vehicle_velocity,np.array(prob<p,dtype=int))
        v_slow = np.amax(np.vstack((v_below-1,slow_c)),axis=0)
        vehicle_velocity = v_slow + v_above
    # 第四步：位置更新
        vehicle_position = vehicle_position + vehicle_velocity
        if flag==1:
            idx_below = np.argwhere(vehicle_velocity<-5)
            vehicle_position[idx_below[0][0]] = last_position
        if vehicle_position[-1] >= road_length:
            temp_pos = vehicle_position[-1] - road_length
            vehicle_position = np.delete(vehicle_position,-1,0)
            #vehicle_position = np.insert(vehicle_position,0,temp_pos,0)
            #temp_vel = vehicle_velocity[-1]
            vehicle_velocity = np.delete(vehicle_velocity,-1,0)
            vehicle_number = vehicle_number - 1
            #vehicle_velocity = np.insert(vehicle_velocity,0,temp_vel,0)
        #cells_data = transfer(vehicle_number,road_length,vehicle_position,vehicle_velocity)
        #data = np.append(data,[cells_data],axis = 0)
        vehicle_distribution = np.zeros((road_length),dtype = int) #存储车辆速度
        for pos in vehicle_position:
            vehicle_distribution[int(pos)] = 255
        log = np.vstack([log,vehicle_distribution.copy()])
    pd.DataFrame(log.T).to_csv('./automata仿真结果.csv')
    fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
    ax1 = fig.add_subplot(1,1,1)

    ax1.set_title('slow_truck_automata')            
    ax1.set_xlabel('元胞')                    
    ax1.set_ylabel('时间/s')                    

    sns.heatmap(data=log, vmin = 0,ax = ax1,vmax = 19, cmap='rainbow') 

    ax1.xaxis.grid(True, which='major')      
    ax1.yaxis.grid(True, which='major')      
    ax1.invert_yaxis()

    #plt.savefig(str(i*50)+'m '+'.jpg',dpi=400,bbox_inches='tight')   
    plt.show() 

    print('finish')
# ####################### 计算密度、流量 ################################
#     for i in range(1500,2000):
#         velocity_avg += np.sum(data[i,1,:])/vehicle_number
#     velocity_avg = velocity_avg/500
#     velocities = np.append(velocities,velocity_avg)
#     del data

# ############################# 画图 ################################### 
# plt.plot(densities,np.multiply(densities,velocities),'*')
# plt.title('Flow-Density Diagram')
# plt.xlabel('Denstity')
# plt.ylabel('Flow')

# print('finish')
#创建first提交