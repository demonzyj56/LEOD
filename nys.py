import numpy as np

def nys(G, k, m, idx=None, mode='normal'):
    n = G.shape[0]

    # do some checking
    assert n > m
    assert m >= k
    assert m > 0
    assert k > 0

    if idx is None:
        idx = np.random.permutation(n)

    cols = idx[:m]
    if mode == 'normal':
        C = G[:, cols]
    else:
        # for memory efficient
        C = G
    W = C[cols, :]

    vec, lamb, vec_T = np.linalg.svd(W, full_matrices=False)
    V = vec[:, :k]
    D = lamb[:k]

    U = np.dot(C, np.sqrt(m/n) * np.dot(V, np.diag(1./D)))
    D = n / m * np.diag(D)
    return U, D

def rsvd(W, k, p=5, q=2):
    m = W.shape[0]
    G = np.random.randn(m, k+p)
    Y = np.dot(W, G)
    for i in range(1, q):
        Y = np.dot(W, Y)
    Q, _ = np.linalg.qr(Y, mode='reduced')
    del _
    B = np.dot(Q.T, np.dot(W, Q))
    vec, lamb, vec_T = np.linalg.svd(B, full_matrices=False)
    V = vec[:, :k]
    D = np.diag(lamb[:k])
    U = np.dot(Q, V)
    return U, D

def rsvd_nys(U, D, k, p=5, q=2):
    m = U.shape[0]
    G = np.random.randn(m, k+p)
    Y = np.dot(U, np.dot(D, np.dot(U.T, G)))
    for i in range(1, q):
        Y = np.dot(U, np.dot(D, np.dot(U.T, Y)))
    Q, _ = np.linalg.qr(Y, mode='reduced')
    B = np.dot(Q.T, np.dot(U, np.dot(D, np.dot(U.T, Q))))
    vec, lamb, vec_T = np.linalg.svd(B, full_matrices=False)
    V = vec[:, :k]
    DD = np.diag(lamb[:k])
    UU = np.dot(Q, V)
    return UU, DD

def rnys(G, k, m, idx=None, p=5, q=2, mode='normal'):
    n = G.shape[0]

    # do some checking
    assert n > m
    assert m >= k
    assert m > 0
    assert k > 0

    if idx is None:
        idx = np.random.permutation(n)

    cols = idx[:m]
    if mode == 'normal':
        C = G[:, cols]
    else:
        # for memory efficient
        C = G
    W = C[cols, :]

    V, D = rsvd(W=W, k=k, p=p, q=q)

    U = np.dot(C, np.sqrt(m / n) * V)
    D = n / m * np.diag(1. / np.diag(D))

    return U, D

def approx_inv(lamb, U, D):
    n = U.shape[0]
    L = np.dot(U, np.diag(np.sqrt(np.diag(D))))
    k = L.shape[1]
    a_inv = 1./lamb * (np.eye(n) - np.dot(np.dot(L, np.linalg.pinv(lamb * np.eye(k) + np.dot(L.T, L))), L.T))
    return a_inv

def approx_inv_b(lamb, U, D, b):
    L = np.dot(U, np.diag(np.sqrt(np.diag(D))))
    k = L.shape[1]
    if lamb != 0:
        a_inv_b = 1. / lamb * (b - np.dot(L, np.dot(np.linalg.pinv(lamb * np.eye(k) + np.dot(L.T, L)), np.dot(L.T, b))))
    else:
        a_inv_b = np.linalg.pinv(U.dot(D.dot(U.T))).dot(b)
    return a_inv_b

def approx_inv_b2(lamb, U, D, b):
    k = U.shape[1]
    a_inv_b = 1. / lamb * (b - np.dot(U.dot(np.linalg.pinv(lamb*np.eye(k) + D.dot(U.T.dot(U)))), D.dot(U.T.dot(b))))
    return a_inv_b

def rnys_efficient(W, k, m, n, p=5, q=2):


    V, D = rsvd(W=W, k=k, p=p, q=q)

    D = n / m * np.diag(1. / np.diag(D))

    return V, D