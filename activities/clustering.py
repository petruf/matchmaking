import numpy as np

def eqsc(D, K=None, G=None):
    "equal-size clustering based on data exchanges between pairs of clusters"
    def error(K, m, D):
        """return average distances between data in one cluster, averaged over all clusters"""
        E = 0
        for k in range(K):
            i = np.where(m == k)[0] # indeces of datapoints belonging to class k
            E += np.mean(D[np.meshgrid(i,i)])
        return E / K
    np.random.seed(0) # repeatability
    N, n = D.shape
    if G is None and K is not None:
        G = N // K # group size
    elif K is None and G is not None:
        K = N // G # number of clusters
    else:
        raise Exception('must specify either K or G')
    m = np.random.permutation(N) % K # initial membership
    E = error(K, m, D)
    t = 1
    while True:
        E_p = E
        for a in range(N): # systematically
            for b in range(a):
                m[a], m[b] = m[b], m[a] # exchange membership
                E_t = error(K, m, D)
                if E_t < E:
                    E = E_t
                    print("{}: {}<->{} E={}".format(t, a, b, E))
                    #plt.clf()
                    #for i in range(N):
                        #plt.text(X[i,0], X[i,1], m[i])
                    #writer.grab_frame()
                else:
                    m[a], m[b] = m[b], m[a] # put them back
        if E_p == E:
            break
        t += 1           
    return m
