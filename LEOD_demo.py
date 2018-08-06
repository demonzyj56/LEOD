import numpy as np
from nys import nys, approx_inv, rnys, rnys_efficient, approx_inv_b, rsvd_nys, approx_inv_b2
from sklearn.neighbors import NearestNeighbors
from knn_by_tree import knn_by_tree

# -------------------------------some aux. functions-----------------------------------------
def index_div(batch_size, idx, m):
    nb_batch = np.int(np.ceil(len(idx) / batch_size))
    sampled_idx_per_batch_K = [[] for i in range(nb_batch)]
    sampled_idx_per_batch = [[] for i in range(nb_batch)]
    rest_idx_per_batch = [[] for i in range(nb_batch)]
    rela_rest_idx_per_batch = [[] for i in range(nb_batch)]
    sorted_idx_m = np.sort(idx[:m])
    K_idx = np.argsort(idx[:m])
    cout = 0
    for i in range(len(idx)):
        cur_batch_idx = i // batch_size
        if cout < m and i == sorted_idx_m[cout]:
            sampled_idx_per_batch_K[cur_batch_idx].append(K_idx[cout])
            sampled_idx_per_batch[cur_batch_idx].append(sorted_idx_m[cout] % batch_size)
            cout += 1
        else:
            rest_idx_per_batch[cur_batch_idx].append(i)
            rela_rest_idx_per_batch[cur_batch_idx].append(i % batch_size)
    return sampled_idx_per_batch_K, sampled_idx_per_batch, rest_idx_per_batch, rela_rest_idx_per_batch

def find_k_largest(a, k):
    '''
    a: an input array
    k
    :return:
    thr: the value of k_th largest val. (in fact the k+1 th)
    '''
    n = len(a)
    top_k_list = a[:k+1].copy()
    cur_thr_idx = np.argmin(top_k_list)
    cur_thr = top_k_list[cur_thr_idx]
    idx = [i for i in range(k+1)]
    for i in range(k+1, n, 1):
        if a[i] > cur_thr:
            top_k_list[cur_thr_idx] = a[i]
            idx[cur_thr_idx] = i
            cur_thr_idx = np.argmin(top_k_list)
            cur_thr = top_k_list[cur_thr_idx]

    return idx



def fast_constrained_eigen_nys(U, D, b, alpha, v0=None, eps = 1e-6, verbose = False):
    # A is U * D * U.T
    n = U.shape[0]
    U_D = U.dot(D)
    flag = True
    iter = 0
    if v0 is None:
        v0 = np.random.rand(n, 1) * 2 - 1
        v0 = v0 / np.linalg.norm(v0)
    obj = 0.5 * alpha - 0.5*(np.dot(v0.T, alpha*v0-np.dot(U_D, np.dot(U.T, v0)))) - np.dot(b.T, v0)
    old_obj = obj.copy()
    v = v0
    # obj_set = []
    # obj_set.append(obj[0, 0])
    while flag:
        iter += 1
        u = alpha*v - np.dot(U_D, np.dot(U.T, v)) + b
        v = u / np.linalg.norm(u)
        obj = 0.5 * alpha - 0.5 * (np.dot(v.T, alpha * v - np.dot(U_D, np.dot(U.T, v)))) - np.dot(b.T, v)
        # obj_set.append(obj[0, 0])
        if iter > 1 and np.abs((obj-old_obj)/obj)[0, 0] < eps:
            flag = False
        old_obj = obj
    # torch.save(obj_set, 'obj_set.pkl')
    if verbose is True:
        print('{} iter. for coverage'.format(iter))
    return v, obj[0, 0]


def powerMethod_nys(U, D, v0=None, eps = 1e-8, verbose = False):
    n = U.shape[0]
    flag = True
    iter = 0
    if v0 is None:
        v0 = np.random.rand(n, 1) * 2 - 1
        v0 = v0 / np.linalg.norm(v0)
    obj = np.dot(v0.T, np.dot(U, np.dot(D, np.dot(U.T, v0))))
    old_obj = obj.copy()
    v = v0
    while flag:
        iter += 1
        u = np.dot(U, np.dot(D, np.dot(U.T, v)))
        v = u / np.linalg.norm(u)
        obj = np.dot(v.T, np.dot(U, np.dot(D, np.dot(U.T, v))))
        if iter > 1 and np.abs((obj-old_obj)/obj)[0, 0] < eps:
            flag = False
        old_obj = obj
    if verbose is True:
        print('{} iter. for coverage'.format(iter))
    return obj[0, 0]

# def dist_calc(feats1, feats2):
#     return cdist(feats1, feats2, 'euclidean') ** 2

def dist_calc(feats1, feats2):
    nb_data1 = feats1.shape[0]
    nb_data2 = feats2.shape[0]
    omega = np.dot(np.sum(feats1 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data2)))
    omega += np.dot(np.sum(feats2 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data1))).T
    omega -= 2 * np.dot(feats1, feats2.T)
    return omega

def sparse_dot(M, M_idx, C):
    n = len(M)
    m = C.shape[1]
    MC = np.zeros(shape=(n, m))
    for i in range(n):
        MC[i, :] = np.dot(M[i], C[M_idx[i], :])
    return MC


# for space complexity, only require O(n*k), use kernel ridge regression
class leod_mem_efficient_fast():  # for space complexity, only require O(n*k)

    def __init__(self, feats, gamma1=0.01, gamma2=0.01, k=6, kk=100, m=200, idx=None, lamb_min=-100., nys_mode='nys', opt_mode='analytic'):
        '''

        :param feats:
        :param gamma1:
        :param gamma2:
        :param k:
        :param kk:
        :param m:
        :param idx:
        :param lamb_min: additional para. for KRR
        :param batch_size: only used to calculate KNN
        :param nys_mode: 'nys'/'rnys'
        :param opt_mode: 'analytic'/'iter'
        '''
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.lamb_min = lamb_min
        self.feats = feats
        self.opt_mode = opt_mode

        nb_data = feats.shape[0]
        if idx is None:
            idx = np.random.permutation(nb_data)
        max_ele_nb = 10000 * 10000
        batch_size = max_ele_nb // m

        sampled_idx_per_batch_K, sampled_idx_per_batch, rest_idx_per_batch, rela_rest_idx_per_batch = index_div(batch_size, idx, m)

        #  --------------------------estimate sigma of kernel matrix----------------------------------
        self.sigma_2 = dist_calc(feats[idx[:kk], :], feats).sum() / nb_data / kk

        #  ----------------------------find knn--------------------------------------------------------
        ind = [[] for ii in range(nb_data)]
        dist_for_W = [[] for ii in range(nb_data)]
        if feats.shape[1] < 5000:
            if nb_data < 50000:
                # method 1
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', leaf_size=1e6, metric='euclidean', n_jobs=8).fit(feats)
                dists, indices = nbrs.kneighbors(feats)
            else:
                # method 2
                dists, indices = knn_by_tree(feats, k+1, verbose=0)
            for i in range(nb_data):
                cur_idx = indices[i, :]
                dist = dists[i, :]
                ind[i] += list(cur_idx)
                dist_for_W[i] += list(dist)
                for j in range(len(cur_idx)):
                    if cur_idx[j] != i:
                        ind[cur_idx[j]].append(i)
                        dist_for_W[cur_idx[j]].append(dist[j])
            del indices, dists
        # method 3: old
        else:  #  high-dimensinal feature
            cout = 0
            for i in range(nb_data):
                # new way to find knn
                if i % batch_size == 0:
                    cur_batch = dist_calc(feats[cout * batch_size:np.min([(cout + 1) * batch_size, nb_data]), :], feats)
                    cout += 1
                    cur_row = cur_batch[0, :]
                else:
                    cur_row = cur_batch[i % batch_size, :]

                # cur_row = dist_calc(feats[i, :][np.newaxis, :], feats)
                cur_idx = find_k_largest(-1. * np.squeeze(cur_row), k)

                ind[i] += cur_idx
                dist_for_W[i] += list(cur_row[np.array(cur_idx)])
                for j in range(len(cur_idx)):
                    if cur_idx[j] != i:
                        ind[cur_idx[j]].append(i)
                        dist_for_W[cur_idx[j]].append(cur_row[cur_idx[j]])
            del cur_batch, cur_row


        #  --------------------------calculate kernel matrix K and M=K*(I+gamma1*L)*K--------------------------------
        M = []  # store M in a sparse format
        M_idx = []
        self.U_K = np.zeros(shape=(nb_data, kk))
        K_sampled = np.exp(-dist_calc(feats[idx[:m], :], feats[idx[:m], :]) / 2. / self.sigma_2)
        if nys_mode == 'nys':
            raise NotImplementedError
        elif nys_mode == 'rnys':
            V_K, self.D_K = rnys_efficient(W=K_sampled, k=kk, m=m, n=nb_data)
        else:
            raise NotImplementedError

        cout = 0
        batch_nb = nb_data // batch_size
        for i in range(nb_data):
            cur_dict = dict(zip(ind[i], dist_for_W[i]))
            cur_ind = []
            cur_val = []
            for key, val in cur_dict.items():
                cur_ind.append(key)
                cur_val.append(val)
            M_idx.append(cur_ind)

            if i % batch_size == 0:
                # cur_batch = np.exp(-dist_calc(feats[cout*batch_size:np.min([(cout+1)*batch_size, nb_data]), :], feats[idx[:m], :]) / 2. / self.sigma_2)
                if cout == batch_nb:
                    cur_batch = np.zeros((nb_data-batch_nb*batch_size, m))
                else:
                    cur_batch = np.zeros((batch_size, m))
                cur_batch[sampled_idx_per_batch[cout], :] = K_sampled[sampled_idx_per_batch_K[cout], :]
                cur_batch[rela_rest_idx_per_batch[cout], :] = np.exp(-dist_calc(feats[rest_idx_per_batch[cout], :], feats[idx[:m], :]) / 2. / self.sigma_2)
                self.U_K[cout * batch_size:np.min([(cout + 1) * batch_size, nb_data]), :] = np.dot(cur_batch, np.sqrt(m / nb_data) * V_K)
                cout += 1

            cur_W_row = np.zeros(shape=(1, nb_data))
            cur_W_row[0, cur_ind] = np.exp(-np.array(cur_val) / 2. / self.sigma_2)
            cur_W_row[0, i] = 0.
            cur_M_row = -1. * self.gamma1 * cur_W_row
            cur_M_row[0, i] = 1. + self.gamma1 * cur_W_row[0, cur_ind].sum()
            M.append(cur_M_row[0, cur_ind])
            del cur_W_row, cur_ind, cur_M_row

        del ind, cur_batch, dist_for_W
        del K_sampled, sampled_idx_per_batch_K, sampled_idx_per_batch, rest_idx_per_batch, rela_rest_idx_per_batch

        # ---------------------------calculate low rank approx.: U, D for K, M-----------------------------------------

        # new_M: k*k
        new_M = np.dot(self.D_K, np.dot(self.U_K.T, sparse_dot(M=M, M_idx=M_idx, C=np.dot(self.U_K, self.D_K))))
        vec, lamb, _ = np.linalg.svd(new_M)
        del M, M_idx, new_M

        self.U_T, self.D_T = np.dot(self.U_K, vec), np.diag(lamb)

        # initialize other para.
        self.alpha = np.ones((nb_data, 1), dtype=float)/np.sqrt(nb_data)

    @staticmethod
    def compute_soft_label(n, n_p):
        n_n = n - n_p
        cp = np.sqrt(float(n_n) / float(n_p))
        cn = -np.sqrt(float(n_p) / float(n_n))
        return cp, cn

    def max_m(self):
        # f = np.dot(self.K, self.alpha)
        f = np.dot(self.U_K, np.dot(self.D_K, np.dot(self.U_K.T, self.alpha)))
        n = self.U_K.shape[0]
        idx = np.argsort(f, axis=0)
        idx = idx[::-1]
        opt_obj = float('-inf')
        for m in range(1, n, 1):
            idx_m = idx[:m:1]
            cp, cn = self.compute_soft_label(n, m)
            y = cn * np.ones((n, 1), dtype=float)
            y[idx_m] = cp+self.gamma2/m
            obj = np.dot(f.transpose(), y)
            if opt_obj < obj:
                opt_obj = obj
                opt_y = y
                opt_m = m
        return opt_m, opt_y


    def optimize_uocl_krr_nys(self):
        n = self.U_K.shape[0]
        t = 0
        m, y = self.max_m()

        # some pre-defined params.
        max_iter = 3
        # obj_list = []
        while t < max_iter:
            # b = np.dot(self.K, y)
            b = np.dot(self.U_K, np.dot(self.D_K, np.dot(self.U_K.T, y)))
            if self.opt_mode == 'analytic':
                self.alpha = approx_inv_b(lamb=-self.lamb_min, U=self.U_T, D=self.D_T, b=b)
            elif self.opt_mode == 'iter':
                self.alpha = self.constrined_eigen_fast(b=b)
            else:
                raise NotImplementedError

            # solve m and y
            _, y = self.max_m()
            # cur_obj = self.calc_obj(y)
            # obj_list.append(cur_obj[0, 0])
            # torch.save(obj_list, 'overall_obj_cifar.pkl')
            # print(cur_obj)
            t += 1
        y = np.sum(y, axis=1)
        predict = np.ones((self.U_K.shape[0],), dtype=int)
        predict[y < 0] = 0
        self.predict = predict
        del self.U_K, self.U_T, self.D_K, self.D_T
        return predict

    def calc_obj(self, y):
        K_y = self.U_K.dot(self.D_K.dot(self.U_K.T.dot(y)))
        return self.alpha.T.dot(self.U_T.dot(self.D_T.dot(self.U_T.T.dot(self.alpha)))) - 2 * K_y.T.dot(self.alpha) + self.lamb_min * self.alpha.T.dot(self.alpha)

# for space complexity, only require O(n*k), and use new method for contrained eigenval problem
class leod_mem_efficient_basic():

    def __init__(self, feats, gamma1=0.01, gamma2=0.01, k=6, kk=100, m=200, idx=None, nys_mode='nys', opt_mode='iter', est_para_mode='rsvd', eps=1e-6):
        '''

        :param feats:
        :param gamma1:
        :param gamma2:
        :param k:
        :param kk:
        :param m:
        :param idx:
        :param lamb_min: additional para. for KRR
        :param batch_size: only used to calculate KNN
        :param nys_mode: 'nys'/'rnys'
        :param opt_mode: 'analytic'/'iter'
        '''
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.feats = feats
        self.opt_mode = opt_mode
        self.eps = eps

        nb_data = feats.shape[0]
        if idx is None:
            idx = np.random.permutation(nb_data)
        max_ele_nb = 10000 * 10000
        batch_size = max_ele_nb // m

        sampled_idx_per_batch_K, sampled_idx_per_batch, rest_idx_per_batch, rela_rest_idx_per_batch = index_div(batch_size, idx, m)

        #  --------------------------estimate sigma of kernel matrix----------------------------------
        self.sigma_2 = dist_calc(feats[idx[:kk], :], feats).sum() / nb_data / kk

        #  ----------------------------find knn--------------------------------------------------------
        ind = [[] for ii in range(nb_data)]
        dist_for_W = [[] for ii in range(nb_data)]
        if feats.shape[1] < 5000:
            if nb_data < 50000:
                # method 1
                nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree', leaf_size=1e6, metric='euclidean', n_jobs=8).fit(feats)
                dists, indices = nbrs.kneighbors(feats)
            else:
                # method 2
                dists, indices = knn_by_tree(feats, k+1, verbose=0)
            for i in range(nb_data):
                cur_idx = indices[i, :]
                dist = dists[i, :]
                ind[i] += list(cur_idx)
                dist_for_W[i] += list(dist)
                for j in range(len(cur_idx)):
                    if cur_idx[j] != i:
                        ind[cur_idx[j]].append(i)
                        dist_for_W[cur_idx[j]].append(dist[j])
            del indices, dists
        # method 3: old
        else:  #  high-dimensinal feature
            cout = 0
            for i in range(nb_data):
                # new way to find knn
                if i % batch_size == 0:
                    cur_batch = dist_calc(feats[cout * batch_size:np.min([(cout + 1) * batch_size, nb_data]), :], feats)
                    cout += 1
                    cur_row = cur_batch[0, :]
                else:
                    cur_row = cur_batch[i % batch_size, :]

                # cur_row = dist_calc(feats[i, :][np.newaxis, :], feats)
                cur_idx = find_k_largest(-1. * np.squeeze(cur_row), k)

                ind[i] += cur_idx
                dist_for_W[i] += list(cur_row[np.array(cur_idx)])
                for j in range(len(cur_idx)):
                    if cur_idx[j] != i:
                        ind[cur_idx[j]].append(i)
                        dist_for_W[cur_idx[j]].append(cur_row[cur_idx[j]])
            del cur_batch, cur_row


        #  --------------------------calculate kernel matrix K and M=K*(I+gamma1*L)*K--------------------------------
        M = []  # store M in a sparse format
        M_idx = []
        self.U_K = np.zeros(shape=(nb_data, kk))
        K_sampled = np.exp(-dist_calc(feats[idx[:m], :], feats[idx[:m], :]) / 2. / self.sigma_2)
        if nys_mode == 'nys':
            raise NotImplementedError
        elif nys_mode == 'rnys':
            V_K, self.D_K = rnys_efficient(W=K_sampled, k=kk, m=m, n=nb_data)
        else:
            raise NotImplementedError

        cout = 0
        batch_nb = nb_data // batch_size
        for i in range(nb_data):
            cur_dict = dict(zip(ind[i], dist_for_W[i]))
            cur_ind = []
            cur_val = []
            for key, val in cur_dict.items():
                cur_ind.append(key)
                cur_val.append(val)
            M_idx.append(cur_ind)

            if i % batch_size == 0:
                # cur_batch = np.exp(-dist_calc(feats[cout*batch_size:np.min([(cout+1)*batch_size, nb_data]), :], feats[idx[:m], :]) / 2. / self.sigma_2)
                if cout == batch_nb:
                    cur_batch = np.zeros((nb_data-batch_nb*batch_size, m))
                else:
                    cur_batch = np.zeros((batch_size, m))
                cur_batch[sampled_idx_per_batch[cout], :] = K_sampled[sampled_idx_per_batch_K[cout], :]
                cur_batch[rela_rest_idx_per_batch[cout], :] = np.exp(-dist_calc(feats[rest_idx_per_batch[cout], :], feats[idx[:m], :]) / 2. / self.sigma_2)
                self.U_K[cout * batch_size:np.min([(cout + 1) * batch_size, nb_data]), :] = np.dot(cur_batch, np.sqrt(m / nb_data) * V_K)
                cout += 1

            cur_W_row = np.zeros(shape=(1, nb_data))
            cur_W_row[0, cur_ind] = np.exp(-np.array(cur_val) / 2. / self.sigma_2)
            cur_W_row[0, i] = 0.
            cur_M_row = -1. * self.gamma1 * cur_W_row
            cur_M_row[0, i] = 1. + self.gamma1 * cur_W_row[0, cur_ind].sum()
            M.append(cur_M_row[0, cur_ind])
            del cur_W_row, cur_ind, cur_M_row

        del ind, cur_batch, dist_for_W
        del K_sampled, sampled_idx_per_batch_K, sampled_idx_per_batch, rest_idx_per_batch, rela_rest_idx_per_batch

        # ---------------------------calculate low rank approx.: U, D for K, M-----------------------------------------

        # new_M: k*k
        new_M = np.dot(self.D_K, np.dot(self.U_K.T, sparse_dot(M=M, M_idx=M_idx, C=np.dot(self.U_K, self.D_K))))
        vec, lamb, _ = np.linalg.svd(new_M)
        del M, M_idx, new_M

        self.U_T, self.D_T = np.dot(self.U_K, vec), np.diag(lamb)

        # torch.save(self.U_T, 'U.pkl')
        # torch.save(self.D_T, 'D.pkl')

        # estimate the parameter for fast solution of constrained eigenvalue
        if est_para_mode == 'power':
            self.a = powerMethod_nys(self.U_T, self.D_T, verbose=True)
        elif est_para_mode == 'rsvd':
            _, D = rsvd_nys(self.U_T, self.D_T, k=kk)
            self.a = np.max(np.diag(D))
        elif est_para_mode == 'trace':
            self.a = self.U_T.dot(np.diag(np.sqrt(np.diag(self.D_T))))**2
            self.a = self.a.sum()
        elif est_para_mode == 'mem_test':
            self.a = 100.
        else:
            raise NotImplementedError

        # initialize other para.
        self.alpha = np.ones((nb_data, 1), dtype=float)/np.sqrt(nb_data)

    @staticmethod
    def compute_soft_label(n, n_p):
        n_n = n - n_p
        cp = np.sqrt(float(n_n) / float(n_p))
        cn = -np.sqrt(float(n_p) / float(n_n))
        return cp, cn

    def max_m(self):
        # f = np.dot(self.K, self.alpha)
        f = np.dot(self.U_K, np.dot(self.D_K, np.dot(self.U_K.T, self.alpha)))
        n = self.U_K.shape[0]
        idx = np.argsort(f, axis=0)
        idx = idx[::-1]
        opt_obj = float('-inf')
        for m in range(1, n, 1):
            idx_m = idx[:m:1]
            cp, cn = self.compute_soft_label(n, m)
            y = cn * np.ones((n, 1), dtype=float)
            y[idx_m] = cp+self.gamma2/m
            obj = np.dot(f.transpose(), y)
            if opt_obj < obj:
                opt_obj = obj
                opt_y = y
                opt_m = m
        return opt_m, opt_y


    def optimize_uocl_nys(self):
        n = self.U_K.shape[0]
        t = 0
        m, y = self.max_m()

        # some pre-defined params.
        max_iter = 3

        while t < max_iter:
            # b = np.dot(self.K, y)
            b = np.dot(self.U_K, np.dot(self.D_K, np.dot(self.U_K.T, y)))

            self.alpha, _ = fast_constrained_eigen_nys(U=self.U_T, D = self.D_T, alpha=self.a, b=b, verbose=False, eps=self.eps)

            # fac = self.alpha**2
            # self.alpha = self.alpha / np.sqrt(fac.sum())

            # solve m and y
            _, y = self.max_m()
            t += 1
        y = np.sum(y, axis=1)
        predict = np.ones((self.U_K.shape[0],), dtype=int)
        predict[y < 0] = 0
        self.predict = predict
        del self.U_K, self.U_T, self.D_K, self.D_T
        return predict



