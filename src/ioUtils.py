# coding=utf-8

import os
import json
import codecs


def readfile2list(filename, encoding="utf-8"):
    with codecs.open(filename, encoding=encoding) as rf:
        return rf.readlines()


def get_dir_file_list(dir, afterfix=""):
    res_list = []
    for root, dir, files in os.walk(dir):
        if afterfix:
            for file in files:
                if file.endswith(afterfix):
                    res_list.append(os.path.join(root, file))
        else:
            res_list.append([os.path.join(root, file) for file in files])

    return res_list


def get_json_att(content, attribute):
    if isinstance(content, str):
        json_dict = json.loads(content)
        return json_dict.get(attribute, None)
    elif isinstance(content, list):
        dict_list = [json.loads(string, attribute)for string in content]
        return [content for content in dict_list if content is not None]