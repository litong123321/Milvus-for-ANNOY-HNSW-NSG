
import json
import os
import tensorflow as tf
import argparse
from loguru import logger
from typing import Text
import json
import re

# # 修改当前工作目录
# os.chdir("ai_platform-master/my_rasa_demo")
import sys

sys.path.append('../')
print(sys.path)
print(tf.__version__)
import numpy as np

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, time):
            return obj.__str__()
        else:
            return super(NpEncoder, self).default(obj)
from rasa.nlu.training_data import load_data
from rasa.nlu import config
from rasa.nlu.model import Trainer, Interpreter
import time
model_dir = './model/raw_data-dcnn/nlu/'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
interpreter = Interpreter.load(model_dir)  # 载入训练后的模型，定义预测器Interpreter对象
while True:
    x = input("说点什么：")
    if x != "EOS":
        print(x)
        print(type(x))
        tic = time.time()
        ans = interpreter.parse(x)
        print(time.time() - tic)
    else:
        break
    print(json.dumps(ans, indent=4, ensure_ascii=False,cls=MyEncoder))