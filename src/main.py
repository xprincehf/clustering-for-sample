# coding=utf-8

import os
import sys
import argsParse
import cluster


if __name__ == '__main__':
    parser = argsParse.args_parse()
    argv = [arg.decode("utf-8") for arg in sys.argv]
    config = parser.parse_args(argv)

    cluster.base_cluster(config)