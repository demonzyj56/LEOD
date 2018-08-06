import time
from LEOD_demo import leod_mem_efficient_fast, leod_mem_efficient_basic
import torch
import numpy as np


# -------------------define some basic functions------------------
def normalize(dataset):
    'A helper function that normalizes each col. of features into the interval [0,1]'
    min_val, _ = torch.min(dataset, 0)
    tmp = dataset.add(-1.0, min_val.repeat(dataset.size(0), 1))
    max_val, _ = torch.max(tmp, 0)
    max_val.add_(0.00000001)
    return tmp / max_val.repeat(dataset.size(0), 1)


# -------------------basic setting------------------------------
seed = 0
torch.manual_seed(seed)  # reproducible
np.random.seed(seed)


dsName = 'mnist'
method = ‘LEOD’
data_dir = 'data/'
outlierRatioList = range(1, 7, 1)

# -------------------loading----------------------------------
classList = range(0, 10, 1)
train_set = torch.load(data_dir+'{}_pretrained.pkl'.format(dsName))
labels = torch.load(data_dir+'{}_pretrained_labels.pkl'.format(dsName))



results = []
for r in outlierRatioList:
    mean_precision = 0.0
    mean_recall = 0.0
    mean_f1 = 0.0
    mean_t = 0.0
    outlier_ratio = r / 10.
    for cur_class in classList:
        inlierSet = train_set[torch.LongTensor([x for x in range(train_set.size(0))])[torch.eq(labels, cur_class)]]
        outlierSet = train_set[torch.LongTensor([x for x in range(train_set.size(0))])[torch.ne(labels, cur_class)]]

        nb_inlier = inlierSet.size(0)
        nb_outlier = int(np.floor(nb_inlier * outlier_ratio / (1 - outlier_ratio)))
        shuffle_idx = torch.randperm(outlierSet.size(0))
        outliers = outlierSet[shuffle_idx[:nb_outlier]]

        noisyData = torch.cat([inlierSet, outliers], 0)
        noisyLabel = torch.cat([torch.ones(inlierSet.size(0), 1), torch.zeros(outliers.size(0), 1)], 0)
        shuffle_idx = torch.randperm(noisyData.size(0))
        noisyData = noisyData[shuffle_idx]
        noisyLabel = noisyLabel[shuffle_idx]
        del inlierSet, outlierSet
        features = normalize(noisyData)  # normalization
        del noisyData



        if method == ‘LEOD’:  # LEOD
            start = time.time()
            feats = features.numpy()
            gamma1 = 1.0
            gamma2 = 1.0
            k = 6
            nys_mode = 'rnys'
            opt_mode = 'basic'

            lamb = -1e0
            m_coef = 10
            l_coef = 20
            m = np.int(features.size(0) / m_coef)
            kk = np.int(features.size(0) / l_coef)

            if opt_mode == 'fast':
                # ——————LEOD-fast--------------------------------------
                clas = leod_mem_efficient_fast(feats=feats, gamma1=gamma1, gamma2=gamma2, k=k, kk=kk,
                                             lamb_min=lamb, m=m, nys_mode=nys_mode)
                predict = clas.optimize_uocl_krr_nys()
            elif opt_mode == 'basic':
                # —————LEOD-basic--------------------
                clas = leod_mem_efficient_basic(feats=feats, gamma1=gamma1, gamma2=gamma2, k=k, kk=kk, m=m,
                                             nys_mode=nys_mode, est_para_mode='rsvd')
                predict = clas.optimize_uocl_nys()
            else:
                raise NotImplementedError
            t = time.time() - start

        else:
            raise NotImplementedError

        # --------------------------------------eval----------------------------------------------------
        noisy_labels = noisyLabel.numpy()

        TP = 0.0
        FP = 0.0
        TN = 0.0
        FN = 0.0
        for i in range(predict.shape[0]):
            if (noisy_labels[i] == 1):
                if (predict[i] == 1):
                    TP += 1
                else:
                    FN += 1
            else:
                if (predict[i] == 1):
                    FP += 1
                else:
                    TN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        mean_precision += precision
        mean_recall += recall
        mean_f1 += f1
        mean_t += t
        # print("Results for digit %d Precision: %f Recall: %f F1: %f" % (test_digits, precision, recall, f1))

    print("Average results for outlier ratio %f: Precision: %f Recall: %f F1: %f Time: %f" % (outlier_ratio,
        mean_precision / len(classList), mean_recall / len(classList), mean_f1 / len(classList),
        mean_t / len(classList)))

    results.append((mean_precision / len(classList), mean_recall / len(classList), mean_f1 / len(classList),
                    mean_t / len(classList)))


