import numpy
from sklearn.neighbors import NearestNeighbors
import scipy.special as sp
import time

from sklearn.neighbors import NearestNeighbors
import numpy as np

status_dict = {
    'filler':2,
    'undefined': -2,
    'completed': -1,
    'in_work': 1,
    'false':0}

def p_density(values,k):
    volume = np.pi ** (k/2) / sp.gamma(k/2 +1)
    density = k/(values * volume)
    return density

def significant(cluster,thresh):
    max_val = np.max(cluster)
    min_val = np.min(cluster)
    res = abs(max_val-min_val)
    return res >= thresh

def sort_indices(values, neighbors):
    new_idx = np.argsort(values)
    idx_dict = {old: new for old, new in zip(range(len(values)), new_idx)}
    new_neighbors = np.array([[idx_dict[x] for x in y] for y in neighbors])
    final_sort = np.argsort(new_neighbors[:,0])
    new_neighbors = new_neighbors[final_sort]
    new_values = values[new_idx]
    return new_values, new_neighbors


class WishartWHeight(object):
    def __init__(self,feature_list,k,h):
        #uninit_clusters = np.full([len(feature_list),-1])
        #start_status = np.full([len(feature_list),-2])
        #self.feature_list = np.stack([uninit_clusters,start_status])
        self.k = k
        self.h = h
        self.open_clusters = {}
        self.ready_clusters = []
        self.cur_cls = 1
        start_time = time.time()
        radii,indices = self.knn(feature_list)
        sort_st_time = time.time()
        print('KNN finished,time elapsed = {:.2f} s'.format(sort_st_time-start_time))
        radii,indices = sort_indices(radii,indices)
        print('sorting_finished ,time elapsed = {:.2f} s'.format(time.time() - sort_st_time))
        self.values = p_density(radii,self.k)
        self.left_right_split(indices)
        print('initialization finished. total time elapsed: {:.2f} s').format(time.time()-start_time)


    def knn(self,x,algorithm = 'kd_tree'):
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
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm=algorithm).fit(x)
        distances, indices = nbrs.kneighbors(x)
        return distances[:,-1], indices

    def left_right_split(self,indices):
        assert type(indices) is np.ndarray, \
        "expected array on input"
        print(indices.shape)
        self.vert_clusters = np.full((indices.shape[0], 1), 0)
        self.status = np.full((indices.shape[0],1),status_dict['undefined'])
        filler = np.full([1,1],status_dict['filler'])
        filter_array = np.less(indices[:,1:],np.expand_dims(indices[:,0],1))
        print(filter_array.shape)
        status = np.where(filter_array,self.status,filler)
        self.left_side = np.stack([indices[:,1:],status])
        print(self.left_side.shape)

    def build_clusters(self):
        for idx in xrange(len(self.values)):
            if idx % 10000 == 0:
                print(idx)
            act_verts = np.not_equal(self.left_side[1][idx],2)
            neighbor_status = self.left_side[0][idx][act_verts]
            if neighbor_status.size == 0:
                self.open_clusters[self.cur_cls] = [idx]
                self.vert_clusters[idx] = self.cur_cls
                self.status[idx] = status_dict['in_work']
                self.cur_cls+=1
                continue
            classes = self.vert_clusters[neighbor_status]
            updated_status = self.status[neighbor_status]
            neighbor_status = np.stack([classes,updated_status])
            neighbor_clusters = set(neighbor_status[0].flat)
            if ((len(neighbor_clusters)==1) or
                (len(neighbor_clusters)==2) and (0 in neighbor_clusters)):
                if np.any(np.equal(neighbor_status[1],status_dict['in_work'])):
                    #TODO fix try-except for checking zero in two-element set
                    try:
                        neighbor_clusters.remove(0)
                    except:
                        pass
                    target = neighbor_clusters.pop()
                    self.status[idx] = status_dict['in_work']
                    self.vert_clusters[idx] = target
                    self.open_clusters[target].append(idx)
                    continue
                else:
                    self.status[idx] = status_dict['completed']
                    self.vert_clusters[idx] = 0
                    continue
            else:
                if np.all(np.equal(neighbor_status[1],status_dict['completed'])):
                    self.status[idx] = status_dict['completed']
                    self.vert_clusters[idx] = 0
                elif (len(self.ready_clusters) > 1) or (self.vert_clusters[0] == 0):
                    self.status[idx] = status_dict['completed']
                    self.vert_clusters[idx] = 0
                    # TODO delete normally-work around several repeated neighbors
                    for x in neighbor_status.T[0]:
                        if x[1] == status_dict['in_work']:
                            try:
                                cluster = np.array(self.open_clusters[x[0]])
                            except:
                                continue
                            if significant(self.values[cluster],self.h):
                                self.ready_clusters.append(cluster)
                            del self.open_clusters[x[0]]
                            self.status[cluster] = status_dict['completed']
                # need to rewrite this part, left for speed evaluation purposes
                else:
                    for x in neighbor_status.T[0]:
                        self.status[idx] = status_dict['completed']
                        self.vert_clusters[idx] = 0
                        if x[1] == status_dict['in_work']:
                            try:
                                cluster = np.array(self.open_clusters[x[0]])
                            except:
                                continue
                            cluster = np.array(self.open_clusters[x[0]])
                            if significant(self.values[cluster],self.h):
                                self.ready_clusters.append(cluster)
                            del self.open_clusters[x[0]]
                            self.status[cluster] = status_dict['completed']
        for open_cluster in self.open_clusters:
            cluster = np.array(open_cluster)
            if significant(self.values[cluster], self.h):
                self.ready_clusters.append(cluster)







"""
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
"""


