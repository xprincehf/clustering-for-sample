# coding=utf-8

import sys
import os

from datetime import datetime
from datetime import timedelta
import random
import time

import argsParse
from cluster import *
from ioUtils import *
from log import *

if __name__ == '__main__':
    parser = argsParse.args_parse()
    argv = [arg.decode("utf-8") for arg in sys.argv[1:]]
    config = parser.parse_args(argv)

    origin_data = readfile2list(config.source)

    if config.iter == 0:
        iter_time = 1
    else:
        iter_time = config.iter

    estimate_iter = list()
    for i in range(iter_time):
        random.shuffle(origin_data)
        attribute_values = get_json_att(origin_data, config.field)
        logging.info("the %s/%s time for cluster" % (i + 1, iter_time))
        start_time = datetime.now()

        cluster = base_cluster(config, attribute_values)
        label = cluster.labels_
        used_time = round(timedelta.total_seconds(datetime.now() - start_time), 2)
        logging.info("finish the cluster, use time %s" % used_time)
        silhouette_score, calinski_harabaz_score = cluster.get_estimate_result()
        logging.info("the cluster estimate is %s and %s" % (silhouette_score, calinski_harabaz_score))

        result_path = os.path.join(config.target, "cluster", config.cluster)
        if config.iter != 0:
            result_path = result_path + "_" + str(i + 1)
        write_cluster_result(result_path, origin_data, label, order="label")
        title = config.cluster + "_" + str(i)
        estimate_iter.append([title, used_time, silhouette_score, calinski_harabaz_score, result_path])

    average_time = round(sum([x[1] for x in estimate_iter]) / iter_time, 2)
    average_silhouette_score = round(sum([x[2] for x in estimate_iter]) / iter_time, 4)
    average_calinski_harabaz_score = round(sum([x[3] for x in estimate_iter]) / iter_time, 4)

    estimate_iter.append(["average", average_time, average_silhouette_score, average_calinski_harabaz_score])
    writelist2file(os.path.join(config.target, "cluster_estimate"), estimate_iter)
