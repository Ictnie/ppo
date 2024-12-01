# 模仿学习和状态转移一起使用，因为需要保存，所以一起调用

from datetime import datetime
import os
import numpy as np
from bc import bc_learn
from transition import trans_learn
current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(current_time)

file = np.load('expert-20241130-22:05:37.npy')
num=200
bc_learn(current_time,file,num)
trans_learn(current_time,file,num)
