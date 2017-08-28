# coding=utf-8

import re

import jieba

PATTERN_NUMBER = re.compile(r'(\d+)(\.\d*)?')
PATTERN_DATE_1 = re.compile(r'\d{2,4}[-/\.]\d{1,2}([-/\.]\d{1,2})?')
PATTERN_DATE_2 = re.compile(ur'\d{2,4}年(\d{1,2}月(\d{1,2}日)?)?')
PATTERN_TIME = re.compile(r'([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?')


def get_cut_sentence(sentence):
    """
    get the segment chinese sentence
    @param sentence: str, the sentence
    @return: list, sentence segment result
    """
    if not sentence:
        return list(sentence)
    return list(jieba.cut(sentence))


def remove_stop_words(sentence, sw_list, cut=False):
    """
    remove the stop words from sentence according to the given dictionary
    @param sentence: str/list, the sentence
    @param sw_list: list, the stop words list
    @param cut: bool, whether the sentence need to segment
    @return: list
    """
    if cut:
        sentence = get_cut_sentence(sentence)
    else:
        if not isinstance(sentence, list):
            print "the given sentence should be a list otherwise params cut should be True"
            raise Exception

    return [word for word in sentence if word not in sw_list]


def sentence_words_count(sentence_list, cut=False, top_n=0):
    """
    simple function for words count
    @param sentence_list: list, the sentences to count
    @param cut: bool, whether the sentence need to segment
    @param top_n: int, return the top n words
    @return: list, the words count result
    """
    res_dict = dict()
    for sentence in sentence_list:
        if cut:
            sentence = get_cut_sentence(sentence)
        else:
            if not isinstance(sentence, list):
                print "the given sentence %s is not a list otherwise params cut should be True" % sentence
            raise Exception

        for word in sentence:
            res_dict[word] = res_dict.get(word, 0) + 1

    res_list = res_dict.items()
    res_list.sort(key=lambda x: x[1], reverse=True)
    return res_list[:top_n] if top_n else res_list


def get_base_feature_vec(sentence, top_words):
    """
    convert the sentence in to vector for clustering
    @param sentence: str, the sentence
    @param top_words: list, the top n words for the features
    @return: list, the convert vector
    """

    sentence = PATTERN_DATE_1.sub("DATEA", sentence)
    sentence = PATTERN_DATE_2.sub("DATEB", sentence)
    sentence = PATTERN_TIME.sub("TIME", sentence)
    sentence = PATTERN_TIME.sub("NUMBER", sentence)
    sentence = sentence.strip()

    date_1_count = 0
    date_2_count = 0
    time_count = 0
    number_count = 0
    space_count = 0

    sentence = get_cut_sentence(sentence)
    for word in sentence:
        if word == "NUMBER":
            number_count += 1
        elif word == "DATEA":
            date_1_count += 1
        elif word == "DATEB":
            date_2_count += 1
        elif word == "TIME":
            time_count += 1
        elif word == " ":
            space_count += 1

    vec = [number_count, date_1_count, date_2_count, time_count, space_count]
    for word in top_words:
        vec.append(sentence.count(word))

    return vec



