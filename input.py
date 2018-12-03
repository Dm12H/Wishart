import csv
import numpy as np
import time
from wishart import knn

with open('/home/demidov/Downloads/ts10e6.csv','rb') as f:
    reader = csv.reader(f, dialect='excel')
    data = np.array([ x for x in reader])
data = data.reshape([-1,3])
print(type(data))
tic = time.time()
distances , indices = knn(data)
toc = time.time()
print(toc-tic)