# coding=utf-8

import codecs
import json
import os


def check_exist(filepath):
    if not os.path.exists(filepath):
        os.makedirs(filepath)

def readfile2list(filepath, encoding="utf-8"):
    with codecs.open(filepath, 'r', encoding=encoding) as rf:
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
        dict_list = [json.loads(string, attribute) for string in content]
        att_list = [json_dict.get(attribute, None) for json_dict in dict_list]
        return [content for content in att_list if content is not None]


def writelist2file(filepath, content, encoding='utf-8'):
    check_exist(filepath)
    with codecs.open(filepath, 'w', encoding=encoding) as wf:
        for item in content:
            if isinstance(item, list):
                item = "\t".join([str(x) for x in item])
            wf.write(item)
            wf.write("\n")


def writejson2file(filepath, json_content, encoding='utf-8'):
    check_exist(filepath)
    with codecs.open(filepath, 'w', encoding=encoding) as wf:
        for item in json_content:
            if isinstance(item, list) or isinstance(item, tuple):
                item = [json.dumps(x, ensure_ascii=False) for x in item]
                item = "\t".join(item)
            else:
                item = json.dumps(item, ensure_ascii=False)
            wf.write(item)
            wf.write("\n")


def write_cluster_result(filepath, origin_data, cluster_labels, format="json", order="label"):
    data_zip_format = zip(origin_data, cluster_labels)
    if order == "sequence":
        pass
    elif order == "label":
        data_zip_format.sort(key=lambda x: x[1])

    if format == "json":
        writejson2file(filepath, data_zip_format)
    elif format == "text":
        writelist2file(filepath, data_zip_format)
