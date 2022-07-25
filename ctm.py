import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

class ctm():
    def __init__(self):
        #统一以m和s作为单位
        self.total_step = 250 #仿真步长
        self.Q_max_per_lane = 1800/3600 #veh/s
        self.K_jam_per_lane = 120/1000    #veh/m
        self.V_free_per_lane = 60/3.6    #m/s
        self.K_max_per_lane = self.Q_max_per_lane/self.V_free_per_lane

        self.init_rate = 0.8  

        self.time_step = 3  #s
        self.g_num_lane = 1 #假设只有一个车道
        self.l_road = 3000  #m
        self.l_cell = self.V_free_per_lane*self.time_step #每一个cell的长度
        self.num_cell = self.l_road/self.l_cell  +2 #总共的cell个数
              
        if self.num_cell%1 != 0:
            print('错误，路不能被整分')
        else:
            self.num_cell = int(self.num_cell)
            
        self.capacity = np.ones(self.num_cell)*self.K_jam_per_lane*self.l_cell #当前cell最大可以承载的数量
        self.capacity[0] = 999
        self.capacity[-1] = 999    
        
        self.occupancy = np.zeros(self.num_cell) #当前cell的车辆数
        self.occupancy[0] = self.init_rate*self.K_max_per_lane*self.l_cell
        
        self.num_lane = np.ones(self.num_cell)*self.g_num_lane
            
        self.block_position = self.num_cell-1 #阻挡住的位置
        self.release_time = 1 #放开的时间
    def get_Q_in(self,occupancy):
        k = occupancy/self.l_cell
        
        if k <= self.Q_max_per_lane/self.V_free_per_lane:
            q = self.Q_max_per_lane
        else:
            q = -((self.Q_max_per_lane)/(self.K_jam_per_lane - self.K_max_per_lane))*(k-self.K_jam_per_lane)
        return q
    
    def get_Q_out(self,occupancy):
        k = occupancy/self.l_cell #每个元胞内车的平均数
        if k <= self.Q_max_per_lane/self.V_free_per_lane:
            q = (self.Q_max_per_lane/self.K_max_per_lane)*k
        else:
            q = self.Q_max_per_lane
        return q
    
    def set_block(self,position,release_time):
        self.block_position = position
        self.release_time = release_time
        self.capacity[self.block_position]=0
        
    def sim(self):
        self.occupancy_next = np.zeros_like(self.occupancy) #复制得到下一阶段所有的occupancy
        log = np.zeros_like(self.occupancy)
        cap_log = np.zeros_like(self.capacity)

        #假设卡车150时进来，200时出去
        position_in = 30
        position_out = 42
        step_in = 70
        step_out = 110
        truck_v = 18/3.6 
        flag = 0
        # set_time = 80
        # self.block_position = 40
        # self.release_time = 100

        for step in range(0,self.total_step):
            # if step == set_time:
            #     self.capacity[self.block_position] = 0
            # if step ==self.release_time: 
            #     self.capacity[self.block_position] = self.K_jam_per_lane*self.l_cell #容量设置为最大

            if step==step_in:
                self.block_position = position_in
                #self.release_time = release_time
                self.capacity[self.block_position]=0
                flag = 1
            elif step==step_out:
                #self.capacity[self.block_position-1] = self.K_jam_per_lane*self.l_cell 
                self.capacity[0:-1]=self.K_jam_per_lane*self.l_cell
                flag = 0
            
            elif flag==1:
                self.block_position = position_in + int((truck_v*self.time_step*(step-step_in)/self.l_cell)) #得到卡车所处的位置
                self.capacity[self.block_position]=0
                self.capacity[:self.block_position] = self.K_jam_per_lane*self.l_cell #上一个格的容量设为最大
                self.capacity[self.block_position+1:] = self.K_jam_per_lane*self.l_cell #上一个格的容量设为最大

            for i in range(0,self.num_cell):   
                if i == 0 or i == self.num_cell-1:
                    self.occupancy_next[i] = self.occupancy[i]
                    continue

                num_in = min(self.get_Q_out(self.occupancy[i-1])*self.num_lane[i-1]*self.time_step,\
                             self.get_Q_in(self.occupancy[i])*self.num_lane[i]*self.time_step)
                cell_in = max(min(self.occupancy[i-1]*self.num_lane[i-1] , num_in , \
                              self.capacity[i]*self.num_lane[i] - self.occupancy[i]*self.num_lane[i]),0)

                num_out = min(self.get_Q_out(self.occupancy[i])*self.num_lane[i]*self.time_step,\
                              self.get_Q_in(self.occupancy[i+1])*self.num_lane[i+1]*self.time_step)
                cell_out = max(min(self.occupancy[i]*self.num_lane[i] , num_out ,\
                               self.capacity[i+1]*self.num_lane[i+1] - self.occupancy[i+1]*self.num_lane[i+1]),0)

                self.occupancy_next[i] = (self.occupancy[i]*self.num_lane[i]+cell_in-cell_out)/self.num_lane[i]
            self.occupancy = self.occupancy_next.copy()
            log = np.vstack([log,self.occupancy] )
            tlog = log*self.num_lane

            cap_log = np.vstack([cap_log,self.capacity])
            cap_log = cap_log*self.num_lane
        log = pd.DataFrame(log*(10/3)).to_csv('./3.csv')
        return tlog

def pic(start,end,position=40,release_time=100):
    model = ctm()
    if start!=end:
        model.num_lane[start:end] = 3
    #model.set_block(position,release_time)
    result = model.sim()
    result = result[::-1]
    result = np.rot90(result, -1)
    return result

result = pic(start=0,end=0)
    
fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
ax1 = fig.add_subplot(1,1,1)

ax1.set_title('slow_truck_ctm')            
ax1.set_xlabel('时间/3s')                    
ax1.set_ylabel('cell内车辆数/veh')                    

sns.heatmap(data=result, vmin = 0,ax = ax1,vmax = 19, cmap='rainbow') 

ax1.xaxis.grid(True, which='major')      
ax1.yaxis.grid(True, which='major')      
ax1.invert_yaxis()

    #plt.savefig(str(i*50)+'m '+'.jpg',dpi=400,bbox_inches='tight')   
plt.show() 

print('finish')
# end  = 40
# for i in range(0,15):
#     start = end-i
#     result = pic(start,end)
    
#     fig = plt.figure(num=1, figsize=(15, 8),dpi=80)
#     ax1 = fig.add_subplot(2,1,1)

#     ax1.set_title(str(i*50)+'m '+'shock absorber(veh)')            
#     ax1.set_xlabel('时间/6s')                    
#     ax1.set_ylabel('cell内车辆数/veh')                    

#     sns.heatmap(data=result, vmin = 0,ax = ax1,vmax = 17, cmap='rainbow') 

#     ax1.xaxis.grid(True, which='major')      
#     ax1.yaxis.grid(True, which='major')      
#     ax1.invert_yaxis()

#     #plt.savefig(str(i*50)+'m '+'.jpg',dpi=400,bbox_inches='tight')   
#     plt.show() 