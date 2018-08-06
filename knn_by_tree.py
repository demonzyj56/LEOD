from sklearn.neighbors import NearestNeighbors, KDTree, BallTree
from multiprocessing import Pool
import numpy as np

def f(X, tree, i,  k, verbose):
    if verbose > 0:
        print(i)
    dist, ind = tree.query([X[i]], k=k)
    return dist, ind

def f_wrap(args):
    return f(*args)

def knn_by_tree(X, k, verbose=0):
    p = Pool()
    # start = time.time()
    nb = X.shape[0]
    tree = BallTree(X, leaf_size=nb, metric='euclidean')
    para_list = []
    for i in range(nb):
        para_list.append((X, tree, i, k, verbose))
    results = p.map(f_wrap, para_list)
    dists = np.zeros((nb, k))
    indices = np.zeros((nb, k), dtype=int)
    for i in range(len(results)):
        dists[i, :] = results[i][0][0, :]
        indices[i, :] = results[i][1][0, :]
    # print('time cost is {}'.format(time.time()-start))
    return dists, indices