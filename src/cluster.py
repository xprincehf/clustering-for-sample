# coding = utf-8

import logging
from datetime import datetime

import numpy as np
from scipy.spatial.distance import cdist
from sklearn import cluster
from sklearn import metrics

from features import *
from ioUtils import *


def base_cluster(config):

    # load the date
    origin_data = readfile2list(config.source)
    attribute_values = get_json_att(origin_data, config.field)

    start_time = datetime.now()

    pre_process_date = []
    for value in attribute_values:
        value = PATTERN_DATE_1.sub("", value)
        value = PATTERN_DATE_2.sub("", value)
        value = PATTERN_TIME.sub("", value)
        value = PATTERN_NUMBER.sub("", value)
        pre_process_date.append(value)

    top_words = sentence_words_count(pre_process_date, cut=True, top_n=20)

    vec_data = [get_base_feature_vec(sentence, top_words) for sentence in attribute_values]

    if config.cluster == "kmeans":
        kmeans = KmeansCluster(vec_data)
        kmeans.predict()
        print kmeans.kmeans.labels_
        write_cluster_result(config.target+".kmeans", origin_data, kmeans.kmeans.labels_)
    elif config.cluster == "dbscan":
        dbscan = DbscanCluster(vec_data)
        dbscan.predict()
        write_cluster_result(config.target+".dbscan", origin_data, dbscan.dbscan.labels_)
    elif config.cluster == "AP":
        ap = APCluster(vec_data)
        ap.predict()
        write_cluster_result(config.target+".ap", origin_data, ap.AP.labels_)
    elif config.cluster == "Birch":
        birch = BirchCluster(vec_data)
        birch.predict()
        write_cluster_result(config.target, origin_data, birch.birch.labels_)


class KmeansCluster:

    def __init__(self, data, sample_rate=0.3):
        self.data = data
        self.length = len(data)
        self.sample_rate = sample_rate
        self.sample_num = int(self.length * sample_rate)
        self._get_random_sample()


    def _get_random_sample(self):
        rand_index = np.random.choice(self.length, self.sample_num, replace=False)
        rand_index = list(rand_index)
        rand_index.sort()
        self.sample_data = [self.data[value] for value in rand_index]

    def get_best_cluster_num_elbow(self):
        X = np.array(self.sample_data)

        kmeans = cluster.KMeans(n_clusters=1, random_state=0).fit(X)
        mean_distortion = sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
        for k in xrange(2, 20):
            kmeans = cluster.KMeans(n_clusters=k, random_state=0).fit(X)
            cur_mean_distortion = sum(np.min(cdist(X, kmeans.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0]
            logging.info("%s clusters meandistortion is %s" % (k, cur_mean_distortion))
            if (mean_distortion - cur_mean_distortion) / mean_distortion < 0.1:
                k -= 1
                break
            mean_distortion = cur_mean_distortion

        return k

    def get_best_cluster_num_silhouette(self):
        X = np.array(self.sample_data)

        kmeans = cluster.KMeans(n_clusters=1, random_state=0, n_jobs=5).fit(X)
        silhouette_score = metrics.silhouette_score(X, kmeans.labels_,metric='euclidean')
        for k in xrange(2, 20):
            kmeans = cluster.KMeans(n_clusters=k, random_state=0, n_jobs=5).fit(X)
            cur_silhouette_score = metrics.silhouette_score(X, kmeans.labels_,metric='euclidean')
            logging.info("%s clusters meandistortion is %s" % (k, cur_silhouette_score))
            if (silhouette_score - cur_silhouette_score) / silhouette_score < 0.1:
                k -= 1
                break
            silhouette_score = cur_silhouette_score

        return k

    def predict(self, find_best="elbow"):
        if find_best == "elbow":
            k = self.get_best_cluster_num_elbow()
        else:
            k = self.get_best_cluster_num_silhouette()

        X = np.array(self.data)

        self.kmeans = cluster.KMeans(n_clusters=k, random_state=0, n_jobs=5).fit(X)
        return self.kmeans

    def get_estimate_result(self):
        if not self.kmeans:
            print "the kmeans model is not exist, please training first."
            return

        silhouette_score = metrics.silhouette_score(np.array(self.data), self.kmeans.labels_)
        calinski_harabaz_score = metrics.calinski_harabaz_score(np.array(self.data), self.kmeans.labels_)
        
        return silhouette_score, calinski_harabaz_score


class DbscanCluster:
    def __init__(self, data, sample_rate=0.3):
        self.data = data
        self.length = len(data)
        self.sample_rate = sample_rate
        self.sample_num = int(self.length * sample_rate)
        self._get_random_sample()


    def _get_random_sample(self):
        rand_index = np.random.choice(self.length, self.sample_num, replace=False)
        rand_index = list(rand_index)
        rand_index.sort()
        self.sample_data = [self.data[value] for value in rand_index]

    def predict(self, eps=0.3, min_samples=10):
        X = np.array(self.data)

        self.dbscan = cluster.DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return self.dbscan

    def get_estimate_result(self):
        if not self.dbscan:
            print "the dbscan model is not exist, please training first."
            return

        silhouette_score = metrics.silhouette_score(np.array(self.data), self.dbscan.labels_)
        calinski_harabaz_score = metrics.calinski_harabaz_score(np.array(self.data), self.dbscan.labels_)

        return silhouette_score, calinski_harabaz_score


class APCluster:
    def __init__(self, data, sample_rate=0.3):
        self.data = data
        self.length = len(data)
        self.sample_rate = sample_rate
        self.sample_num = int(self.length * sample_rate)
        self._get_random_sample()

    def _get_random_sample(self):
        rand_index = np.random.choice(self.length, self.sample_num, replace=False)
        rand_index = list(rand_index)
        rand_index.sort()
        self.sample_data = [self.data[value] for value in rand_index]

    def predict(self, damping=0.3, preference=None):
        X = np.array(self.data)

        self.AP = cluster.AffinityPropagation(damping=damping, preference=preference)
        return self.AP

    def get_estimate_result(self):
        if not self.AP:
            print "the AP model is not exist, please training first."
            return

        silhouette_score = metrics.silhouette_score(np.array(self.data), self.AP.labels_)
        calinski_harabaz_score = metrics.calinski_harabaz_score(np.array(self.data), self.AP.labels_)

        return silhouette_score, calinski_harabaz_score


class BirchCluster:
    def __init__(self, data, sample_rate=0.3):
        self.data = data
        self.length = len(data)
        self.sample_rate = sample_rate
        self.sample_num = int(self.length * sample_rate)
        self._get_random_sample()

    def _get_random_sample(self):
        rand_index = np.random.choice(self.length, self.sample_num, replace=False)
        rand_index = list(rand_index)
        rand_index.sort()
        self.sample_data = [self.data[value] for value in rand_index]

    def predict(self, threshold=0.3, branching_factor=None):
        X = np.array(self.data)
        if not branching_factor:
            branching_factor = self.length / 20
        self.birch = cluster.Birch(threshold=threshold, branching_factor=branching_factor, n_clusters=None).fit(X)
        return self.birch

    def get_estimate_result(self):
        if not self.birch:
            print "the birch model is not exist, please training first."
            return

        silhouette_score = metrics.silhouette_score(np.array(self.data), self.birch.labels_)
        calinski_harabaz_score = metrics.calinski_harabaz_score(np.array(self.data), self.birch.labels_)

        return silhouette_score, calinski_harabaz_score
