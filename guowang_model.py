import json
import os
import tensorflow as tf
import argparse
# import asyncio
# import logging
from loguru import logger
from typing import Text
import json
import sys
c = os.path.dirname(__file__)
config_file_path = c + '/config_simbert.yml'
print(c)
sys.path.append('../')
sys.path.append(c)
sys.path.append(os.path.dirname(c))
sys.path.append(os.path.dirname(c) + '/text_classfication_demo')
sys.path.append(os.path.dirname(os.path.dirname(c)))
sys.path.append('../')
sys.path.append(c)
sys.path.append(os.path.dirname(c))
sys.path.append(os.path.dirname(c) + '/text_classfication_demo')
sys.path.append(os.path.dirname(os.path.dirname(c)))
sys.path.append('/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/rasa/')
sys.path.append('/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/')
sys.path.append('/home/li/code/en_intent/en_intent/rasa_base_demo/text_classfication_demo/')
from rasa.nlu.config import load
from rasa.nlu.utils.data_proprecess import clean_data
from rasa.nlu.featurizers.bert_featurizer_server import BertFeaturizerServer
from rasa.nlu.classifiers.simbert_match import l2
import re
from annoy import AnnoyIndex



import jieba
# # 修改当前工作目录
# os.chdir("ai_platform-master/my_rasa_demo")
import sys

sys.path.append('../')
sys.path.append(c)
sys.path.append(os.path.dirname(c))
sys.path.append(os.path.dirname(os.path.dirname(c)))

print(sys.path)
print(tf.__version__)

from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer, Interpreter
import time
# from en_intent.sample_content import *
from keras_textclassification.conf.path_config import \
    path_yizhi_multi_news_train,path_yizhi_multi_news_valid,path_byte_multi_news_label
# from keras_textclassification.conf.path_config import path_yizhi_multi_news_train, path_yizhi_multi_news_valid, path_byte_multi_news_label
from sklearn.metrics import classification_report
from sklearn import feature_extraction
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def ifidf_vector(tests):

    result = [jieba.cut('我是中国人', use_paddle=True) for e in tests]

    vectorizer = CountVectorizer()

    X = vectorizer.fit_transform(result)

    # 获取词袋模型中的所有词语
    word = vectorizer.get_feature_names()
    print(len(word))

    # 获取每个词在该行（文档）中出现的次数
    counts = X.toarray()
    print(counts)

    transformer = TfidfTransformer()
    tfidf = transformer.fit_transform(X)

    count_test = vectorizer.transform(['aaa, ddd, ddd, ccc'])
    rr = transformer.transform(count_test)
    rr = rr.toarray()


class IntentRecognizer(object):
    def __init__(self, training_data_path):
        self.annoyIndex = None
        self.datas_map = None
        self.count_vectorizer = None
        self.tfidf_transformer = None
        self.training_data_path = train_data_path
        self.file_config_path = str(training_data_path).split('en_intent')[0] + 'en_intent/my_rasa_demo' + '/config_simbert.yml'
        self.pipeline = '/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/my_rasa_demo/config_simbert.yml'
        self.synonyms = self.pipeline[0].get('synonyms', []) or []
        self.stopwords = self.pipeline[1].get('stopwords', []) or []

        self.defaults = {
            "http_path": "http://121.36.172.209:8126/encode",
            "mode": "0",
            "ip": "192.168.110.8",
            "port": 5555,
            "port_out": 5556,
            "batch_size": 256
        }
        self.bert_encoder = BertFeaturizerServer(self.defaults)

    def rasa_predict_one(self, text_data, get_vector_num=8):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        text_data = clean_data(text_data, stopwords=self.stopwords)
        s_vec = self.bert_encoder.encode('1', [text_data])
        s_vec_l2 = l2(s_vec)[0]
        idx, conf = self.annoyIndex.get_nns_by_vector(s_vec_l2, get_vector_num, include_distances=True)
        idx_label = [self.datas_map[i][1] for i in idx]
        idx_text = [self.datas_map[i][0] for i in idx]
        tmp = []
        tmp_idx = []
        for i in range(len(idx_label)):
            if idx_label[i] not in tmp:
                tmp.append(idx_label[i])
                tmp_idx.append(i)

        intent_ranking = [{"name": idx_label[i], "confidence": conf[i], "sim_sentence": idx_text[i]} for i in
                          tmp_idx]
        intent = intent_ranking[0]
        predict_label = intent.get('name')
        return text, predict_label, intent.get('confidence')


    def rasa_train_2(self, training_data_path, build_tree_num=17):
        # 模型训练
        print('start trainging')
        labels = []
        texts = []
        with open(training_data_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
            for line in lines:
                label, text = line.split('|,|')
                labels.append(label)
                text = clean_data(text, self.synonyms, self.stopwords)
                texts.append(text)


        # bert embedding
        print("bert featurize start!")
        text_embeddings = self.bert_encoder.encode('1', texts)
        print("bert featurize finish!")
        text_embeddings = np.array(text_embeddings)
        text_embeddings_l2 = l2(text_embeddings)
        vec_size = 768


        # tf-idf embedding
        # result = [' '.join(list(jieba.cut(text.strip(), use_paddle=True))) for text in texts]
        # self.count_vectorizer = CountVectorizer()
        # X = self.count_vectorizer.fit_transform(result)
        # # 获取词袋模型中的所有词语
        # words = self.count_vectorizer.get_feature_names()
        # vec_size = len(words)
        # # 获取每个词在该行（文档）中出现的次数
        # counts = X.toarray()
        # print(counts)
        # self.tfidf_transformer = TfidfTransformer()
        # text_embeddings = self.tfidf_transformer.fit_transform(X).todense()
        # text_embeddings = np.array(text_embeddings)
        # text_embeddings_l2 = text_embeddings
        #

        t = AnnoyIndex(vec_size, 'dot')
        for idx, val in enumerate(text_embeddings_l2):
            t.add_item(idx, val)
        t.build(build_tree_num)  # 10 trees
        self.annoyIndex = t
        self.datas_map = list(zip(texts, labels))
        import pickle



        # for build_num in range(16, 17):
        #     t = AnnoyIndex(vec_size, 'dot')
        #     for idx, val in enumerate(text_embeddings_l2):
        #         t.add_item(idx, val)
        #     t.build(build_num)  # 10 trees
        #     self.annoyIndex = t
        #     self.datas_map = list(zip(texts, labels))
        #     for vect_num in range(7, 22):
        #         acc, f_acc = self.rasa_test_2(path_yizhi_multi_news_valid, vect_num)
        #         print(vect_num)
        #         print(acc)
        #         print(f_acc)

    def test_0(self):
        pass

    # p = '/Users/zhouxw/PycharmProjects/yizhi_work/en_intent/my_rasa_demo/online_conversation_data2.txt'
    # online_out_path = os.path.dirname(p) + '/online_output_0.9.txt'
    # out = open(online_out_path, 'w')
    # with open(p, 'r', encoding='utf8') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if len(line.strip().split('\t')) > 1:
    #             l, words = line.strip().split('\t')
    #             if words:
    #                 text = ''.join(words.split(' '))
    #                 text, label, confidence = obj.rasa_predict_one(text)
    #                 if confidence >= 0.9:
    #                     o = '\t'.join([text, label, str(confidence)])
    #                     out.write(o)
    #                     out.write('\n')
    # out.close()

    def rasa_test_2(self, test_data_path, get_vector_num=8):
        print('start testing')
        category_set = set()
        test_list = []
        with open(test_data_path, encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines:
            if line and len(line.split('|,|')) > 1:
                label, text = line.split('|,|')
                category_set.add(str(label))
                test_list.append((text.strip(), label.strip()))
        id_2_label = dict()
        label_2_id = dict()
        for i, ele in enumerate(list(category_set)):
            id_2_label[i] = ele
            label_2_id[ele] = i
        import numpy as np
        # test_list = np.random.permutation(test_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        corr_num = 0
        final_corr_num = 0
        # test_list = test_list
        all_num = len(test_list)
        print_num = 0
        y_idx = []
        pred_y_idx = []
        target_names = list(category_set)
        time_start = time.time()
        for i in range(len(test_list)):
            text, label = test_list[i]
            text = clean_data(text, stopwords=self.stopwords)
            print_num += 1
            # if print_num % 20 == 0:
            #     print(print_num)

            # bert embedding
            s_vec = self.bert_encoder.encode('1', [text])

            # tf-idf embedding
            # text_text = jieba.cut(text, use_paddle=True)
            # count_test = self.count_vectorizer.transform(text_text)
            # s_vec = self.tfidf_transformer.transform(count_test)
            # s_vec = s_vec.toarray()

            s_vec_l2 = l2(s_vec)[0]
            idx, conf = self.annoyIndex.get_nns_by_vector(s_vec_l2, get_vector_num, include_distances=True)
            idx_label = [self.datas_map[i][1] for i in idx]
            lbl_2_num = dict()
            for l in idx_label:
                lbl_2_num[l] = lbl_2_num.get(l, 0) + 1
            lbl_2_num = sorted(lbl_2_num.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
            final_lbl = None
            for k, v in lbl_2_num:
                final_lbl = k
                break
            idx_text = [self.datas_map[i][0] for i in idx]
            tmp = []
            tmp_idx = []
            for i in range(len(idx_label)):
                if idx_label[i] not in tmp:
                    tmp.append(idx_label[i])
                    tmp_idx.append(i)

            intent_ranking = [{"name": idx_label[i], "confidence": conf[i], "sim_sentence": idx_text[i]} for i in
                              tmp_idx]
            intent = intent_ranking[0]
            predict_label = intent.get('name')
            y_idx.append(label_2_id.get(str(label)))
            pred_y_idx.append(label_2_id.get(predict_label))
            # print(final_lbl)
            # print(intent.get('name'))
            if final_lbl == str(label):
                final_corr_num += 1
            if intent.get('name') == str(label):
                corr_num += 1

        acc = (1.0 * corr_num / all_num)
        final_acc = (1.0 * final_corr_num) / all_num
        return acc, final_acc
        # print('---------------------------------- rasa acc: %.4f' % acc)
        # print('---------------------------------- final rasa  acc: %.4f' % final_acc)
        #
        # report_predict = classification_report(y_idx, pred_y_idx,
        #                                        target_names=target_names, digits=9)
        #
        # s = report_predict.strip().split('\n')[-1]
        # s_list = s.split(' ')
        # p, r, f1 = float(s_list[4]), float(s_list[5]), float(s_list[6])
        # return acc, p, r, f1, str(time.time() - time_start)


if __name__ == '__main__':
    preprocessed_data_path = \
        '/home/li/code/en_intent/en_intent/rasa_base_demo/text_classfication_demo/keras_textclassification/data/yizhi/train1.csv'
    test_data_path =  '/home/li/code/en_intent/en_intent/rasa_base_demo/text_classfication_demo/keras_textclassification/data/yizhi/test1.csv'
    train_data_path = os.path.dirname(preprocessed_data_path) + '/train.json'
    # rasa_preprocess(preprocessed_data_path)
    # rasa_train(train_data_path)
    obj = IntentRecognizer(path_yizhi_multi_news_train)
    obj.rasa_train_2(path_yizhi_multi_news_train)
    acc, f_acc = obj.rasa_test_2(path_yizhi_multi_news_valid, 8)
    print(acc)
    print(f_acc)









