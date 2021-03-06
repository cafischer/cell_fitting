from analyze_intracellular.spike_sorting import k_means_clustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering


def cluster(X, method, args):
    if method == 'k_means':
        n_clusters = args.get('n_cluster')
        labels = k_means_clustering(X, n_clusters)
    elif method == 'dbscan':
        eps = args.get('eps')
        min_samples = args.get('min_samples')
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_
    elif method == 'agglomerative':
        n_clusters = args.get('n_cluster')
        linkage = args.get('linkage')
        ward = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        ward.fit(X)
        labels = ward.labels_
    else:
        raise ValueError('Method not implemented!')
    return labels
