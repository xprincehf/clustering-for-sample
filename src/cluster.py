# coding = utf-8

import time
from datetime import datetime
import ioUtils
from features import *
import numpy as np
from numpy import random
from sklearn import cluster


def base_cluster(config):

    # load the date
    origin_data = ioUtils.readfile2list(config.source)
    attribute_values = ioUtils.get_json_att(origin_data, config.filed)

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
        pass
    elif config.cluster == "dbscan":
        pass
    elif config.cluster == "AP":
        pass
    elif config.cluster == "Birch":
        pass


def kmeans_cluster(vector_list):
    input_data = np.array(vector_list)


def dbscan_cluster(vector_list):


def affinity_propagation_cluster(vector_list):


def brich_cluster(vector_list):