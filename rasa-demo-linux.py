import json
import os
import tensorflow as tf
import argparse
# import asyncio
# import logging
from loguru import logger
from typing import Text
import json
import re
c = os.path.dirname(__file__)
from tqdm import tqdm
# # 修改当前工作目录
# os.chdir("ai_platform-master/my_rasa_demo")
import sys
#
# sys.path.append('/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/rasa')
sys.path.append('/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/rasa/')
sys.path.append('/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/')

print(sys.path)
print(tf.__version__)

from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer, Interpreter
import datetime
import time
#from en_intent.sample_content import *

def time_value(dec):
    def wrapper(*args,**kwargs):
        start_time = time.time()
        get_str = dec(*args,**kwargs)
        end_time = time.time()
        print("函数运行共耗时：",end_time-start_time)
        return get_str
    return wrapper

def print_calc_time(func):
    def wrapper(*args, **kw):
        start_time = time.time()
        func(*args, **kw)
        end_time = time.time()
        ss = end_time - start_time
        print('Function <{}> run time is {}s.'.format(func.__name__, ss))
    return wrapper


def train_nlu(
        config_file="config_simbert.yml",
        training_data_file="raw_data.json",
        model_directory: Text = "/model",
        model_name: Text = "current",
):
    training_data = load_data(training_data_file)  # 基于load_data，加载训练数据
    trainer = Trainer(config.load(config_file))  # 基于config.load加载配置文件，并定义Trainer类
    trainer.train(training_data)  # 基于训练数据training_data对每个组件进行训练
    print('---------train done!------------')
    # Attention: trainer.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name)
    print('model_path:', model_path)
    model_directory = trainer.persist(model_path, fixed_model_name="nlu")  # 保存模型文件
    logger.info("Model trained. Stored in '{}'.".format(model_directory))
    return model_directory  # 返回模型保存的目录

@print_calc_time
def train():
    # 模型训练
    config_file = "/home/li/code/en_intent/en_intent/en_intent/rasa_base_demo/my_rasa_demo/config_simbert.yml"  # 该文件中包含了pipeline，及各个component的参数定义

    training_data_file = '/home/li/code/en_intent/en_intent/en_intent/599109455/train.json'
    test_model_directory = train_nlu(config_file=config_file,
                                     training_data_file=training_data_file,
                                     model_directory="./model",
                                     model_name="raw_data-dcnn")

@print_calc_time
def test():
    test_data = '/home/li/code/en_intent/en_intent/en_intent/599109455/test.txt'
    test_list = []
    with open(test_data, encoding="utf-8") as f:
        lines = f.readlines()
    for line in tqdm(lines):
        if not line:
            continue
        if line and len(line.split('\t')) > 1:
            text, label = line.split('\t')[0],line.split('\t')[1]
            test_list.append((text.strip(), label.strip()))
    import numpy as np
    # test_list = np.random.permutation(test_list)
    model_dir = './model/raw_data-dcnn/nlu/'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    interpreter = Interpreter.load(model_dir)  # 载入训练后的模型，定义预测器Interpreter对象
    corr_num = 0
    # test_list = test_list
    all_num = len(test_list)
    print_num = 0
    for i in tqdm(range(len(test_list))):
        text, label = test_list[i]
        print_num += 1
        if print_num % 200 == 0:
            print(print_num)
        result = interpreter.parse(text)
        intent = result.get('intent')
        if intent.get('name') == str(label):
            corr_num += 1
    print(1.0 * corr_num / all_num)

#@print_calc_time
def preprocess():
    train_data = '/home/li/code/en_intent/en_intent/en_intent/599109455/train.txt'
    with open(train_data, encoding="utf-8") as f:
        datas = f.readlines()
    json_data = []
    for i in tqdm(datas):
        try:
            #print(tmp0,tmp1)
            if i and len(i.split("\t"))>1:
                tmp0, tmp1 = i.split("\t")[0],i.split("\t")[1]
                #print(tmp0, tmp1)
                json_data.append({"intent": tmp1.strip(), "text": tmp0.strip()})
        except Exception as e:
            #print(i)
            print(e)
    with open('/home/li/code/en_intent/en_intent/en_intent/train.json', "w", encoding="utf-8") as f:
        json.dump({"rasa_nlu_data": {"common_examples": json_data}}, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':


    preprocess()
    train()
    test()



#0.6522781774580336
#0.6546762589928058
#0.6654676258992805
#clf.best_params_ {'C': 10, 'gamma': 'auto', 'kernel': 'sigmoid'} 0.6762589928057554
# {'knn__algorithm': 'kd_tree', 'knn__leaf_size': 30, 'knn__n_neighbors': 4} 0.6618705035971223
#{'alpha': 0.3} BernoulliNB 0.65107





