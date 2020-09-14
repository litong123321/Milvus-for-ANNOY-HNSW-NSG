from bert_serving.client import BertClient
import numpy as np
#import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import tqdm
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
defaults = {
    "http_path": "http://60.191.10.76:55003/bert/encode",
    "mode": "0",
    "ip": "192.168.110.10",
    "port": 5555,
    "port_out": 5556,
    "batch_size": 256
}
import requests
def httpEncode(texts):
    requests.adapters.DEFAULT_RETRIES = 5  # 增加重连次数
    s = requests.session()
    s.keep_alive = False  # 关闭多余连接
    #s.get(url)  # 你需要的网址
    with s.post("http://60.191.10.76:55003/bert/encode",headers={'Connection': 'close'},
                       json={"id": 123,"texts": texts,"is_tokenized": False}) as r:
        numpy_features = []
        try:
            numpy_features = r.json()["result"]
        except:
            for i in texts:
                with requests.post("http://60.191.10.76:55003/bert/encode", json={
                    "id": 123,
                    "texts": [i],
                    "is_tokenized": False
                }) as r:
                    try:
                        numpy_features.append(r.json()["result"][0])
                    except:
                        print("error data:", "****", i, "****")
    return numpy_features


print(httpEncode(['介绍一下一知荣誉奖项','介绍一下一知荣誉奖项']))

'''
def clientEncode(texts):

    bc = BertClient(defaults["ip"],
                    defaults["port"],
                    defaults["port_out"],
                    check_version=False
                    )
    numpy_features = bc.encode(texts)
    bc.close()
    return numpy_features

print(len(clientEncode(['教授说'])[0]))

TEXT = []
y = []
train_data = 'D:/code/RASA_20200902/en_intent/en_intent/599109455/train.txt'
with open(train_data, encoding="utf-8") as f:
    datas = f.readlines()
    for i in (datas):
        try:
            #print(tmp0,tmp1)
            if i and len(i.split("\t"))>1:
                text,label = i.split("\t")[0],i.split("\t")[1]
                #print(tmp0, tmp1)
                TEXT.append((text))
                y.append(int(label))
        except Exception as e:
            #print(i)
            print(e)


X = clientEncode(TEXT)
# encode class values as integers
encoder = LabelEncoder()
encoded_Y = encoder.fit_transform(y)
# convert integers to dummy variables (one hot encoding)
y = np_utils.to_categorical(encoded_Y)
# define model structure
print(y.shape)#(3085, 366)

model = Sequential()
model.add(Dense(units=768,activation='elu'))
model.add(Dropout(0.3))
model.add(Dense(units=106,activation='softmax'))
    # Compile model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.1, random_state=0)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                              patience=5, min_lr=0.0001)
#model.fit(X_train, Y_train, callbacks=[reduce_lr])
model.fit(X_train, Y_train,epochs=150,batch_size=256,callbacks=[reduce_lr])

TEXT = []
y = []
train_data = 'D:/code/RASA_20200902/en_intent/en_intent/599109455/test.txt'
with open(train_data, encoding="utf-8") as f:
    datas = f.readlines()
    for i in (datas):
        try:
            # print(tmp0,tmp1)
            if i and len(i.split("\t")) > 1:
                text, label = i.split("\t")[0], i.split("\t")[1]
                # print(tmp0, tmp1)
                TEXT.append((text))
                y.append(int(label))
        except Exception as e:
            # print(i)
            print(e)

X_test = clientEncode(TEXT)
encoded_Y = encoder.fit_transform(y)
# convert integers to dummy variables (one hot encoding)
y_test = np_utils.to_categorical(encoded_Y)
score = model.evaluate(X_test, y_test, batch_size=256)
print('---------accuracy:',score[1],'loss:',score[0])
print(X_test.shape,y_test.shape)
#y_test_pred = model.predict(X_test)
#print(y_test_pred)

'''