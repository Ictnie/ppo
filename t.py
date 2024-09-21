import numpy as np
a=np.load('odb.npy')
for i in a:
    obs1=np.array([[i[0],i[1],i[2]]],dtype='float32')
    action=np.array([[i[3]]],dtype='float32')
    obs2=np.array([[i[4],i[5],i[6]]],dtype='float32')
    p=[obs1,action,obs2]
    print(p)