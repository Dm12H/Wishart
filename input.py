import csv
import numpy as np
import time
from wishart import WishartWHeight
with open('/home/demidov/Downloads/ts10e6.csv','rb') as f:
    reader = csv.reader(f, dialect='excel')
    data = np.array([ x for x in reader])
data = data.reshape([-1,3])
print(type(data))
wish = WishartWHeight(data,11,0.01)
tic = time.time()
wish.build_clusters()
toc = time.time()
print('cluster construction took {}'.format(toc-tic))
clusters = wish.ready_clusters

