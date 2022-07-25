from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#从文件中加载数据
def load_data(path):
    with open(path,'rb') as f:
        data = f.read().splitlines()
    length = len(data)
    #变成utf-8编码
    for i in range(length):
        data[i]=data[i].decode('utf-8')
    return length,data

#将数据按0.8：0.1：0.1的比例划分为训练集，验证集，测试集
def split_data(r_len,f_len,r_data,f_data):
    #构造标签
    r_label = [0]*r_len
    f_label = [1]*f_len
    cv = CountVectorizer(analyzer='word',stop_words='english')#创建词袋数据结构
    all_data = r_data + f_data
    all_label = r_label + f_label
    all_label = np.array(all_label)
    cv_fit = cv.fit_transform (all_data) 
    cv_fit = cv_fit.A  #csr稀疏矩阵转为ndarray
    x_train,X_test,y_train,y_test = train_test_split(cv_fit,all_label,test_size = 0.2 ,shuffle = True, random_state = 13)
    x_val,x_test,y_val,y_test = train_test_split(X_test,y_test,test_size = 0.5 ,shuffle = True, random_state = 17)
    return x_train,y_train,x_val,y_val,x_test,y_test

if __name__ == '__main__':
    real_path = './clean_data/clean_real.txt'
    fake_path = './clean_data/clean_fake.txt'
    r_len,r_data = load_data(real_path)
    f_len,f_data = load_data(fake_path)
    x_train,y_train,x_val,y_val,x_test,y_test = split_data(r_len,f_len,r_data,f_data)
    # 利用网格搜索法确定最优的邻居数量
    parameters = {'n_neighbors':[i for i in range(1,15)]}
    neigh = KNeighborsClassifier()
    cvneigh = GridSearchCV(neigh,parameters,cv=5)
    cvneigh.fit(x_train,y_train)
    print(cvneigh.best_params_)
    print(cvneigh.best_score_)
    print(cvneigh.score(x_test,y_test))
    

