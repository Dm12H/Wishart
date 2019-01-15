import csv
import numpy as np
import time
from wishart import WishartWHeight
with open('/home/demidov/Downloads/ts10e6.csv','rb') as f:
    reader = csv.reader(f, dialect='excel')
    data = np.array([ x for x in reader])
data = data.reshape([-1,3])
print(type(data))
wish = WishartWHeight(data,11,1.)
tic = time.time()
wish.build_clusters()
toc = time.time()
print('cluster construction took {:.2f} sec'.format(toc-tic))
clusters = wish.ready_clusters
print('clusters number: {}'.format(len(clusters.keys())))
num_used = 0
for v in clusters.values():
    num_used+=len(v)
print('used points: {:.0%}'.format(num_used/ 1000000.))
print ('merges commenced: {}'.format(wish.merge_counter))

