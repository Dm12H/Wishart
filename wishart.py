import numpy
from sklearn.neighbors import NearestNeighbors
import scipy.special as sp
import timeit

from sklearn.neighbors import NearestNeighbors
import numpy as np

status_dict = {
    'filler':2,
    'undefined': -2,
    'completed': -1,
    'in_work': 1,
    'false':0}

def knn(x,algorithm = 'kd_tree'):
    """
    :param x: input ndarray of floats of shape [N,d1,d2,d3...dm]
              where m is number of dimensions of vector,
              N is total number of samples
    :param algorithm:
    :return:
             distances of shape [N] - radius of hypersphere around point x
             indices - array of indices of shape [N,d1,d2,...,dk,dk+1],
             where k is number of neighbors.(first coord is just index of element.
    """
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='kd_tree').fit(x)
    distances, indices = nbrs.kneighbors(x)
    return distances[:,-1], indices

def sort_indices(values,neighbors):
    new_idx = np.argsort(values)
    idx_dict = {old:new for old,new in zip(range(len(x)),new_idx)}
    new_neighbors = np.array([[idx_dict[x] for x in y] for y in neighbors])
    new_values = values[new_idx]
    return new_values, new_neighbors

def p_density(values,k):
    volume = np.pi ** (k/2) / sp.gamma(k/2 +1)
    density = k/(values * volume)
    return density

def check_significance(cluster,thresh):
    max_val = np.max(cluster.values())
    min_val = np.min(cluster.values())
    res = np.abs(max_val-min_val)
    return res >= thresh

def neighbors_decision(n_list,num):
    mask = np.not_equal(n_list[1],2)
    act_array = np.extract(np.column_stack((mask, mask)),n_list)
    if masked.size == 0:
        return num,False
    elif np.all(np.equal(act_array,act_array[0][0])):
        return act_array[0][0],False
    else:
        if not np.any(np.equal(act_array[1],1)):
            return 0,False
        else:
            return None,True

def reorder_case_1(n_list,cluster_list,thresh):
    count_list = np.equal(n_list[1],1)
    act_clusters = np.select(n_list[0],count_list)
    select_list = np.cumsum(count_list)*count_list
    to_finalize = np.array([False]+[ckeck_significance(cluster_list[x],thresh) for x in act_clusters])
    to_remove = np.array([False]+[not ckeck_significance(cluster_list[x],thresh) for x in act_clusters])
    to_remove_list = np.select(n_list[0], to_remove[select_list])
    to_finalize_list = np.select(n_list[0], to_finalize[select_list])
    return 0, to_remove_list , to_finalize_list

def reorder_case_2(n_list,cluster_list,thresh):
    


#def wishart_graph(density,neighbors):


