# rasa

rasa是一个对话机器人的框架，组成包括rasa core 和 rasa nlu。

**rasa nlu负责意图理解；rasa core负责对话管理**。

我们现在的对话系统使用到的部分只有rasa nlu部分。我们负责意图理解（无上下文）,技术部负责对话管理。

以API的方式提供给一知大脑或者是探意服务。

先介绍一下**rasa nlu**部分的结构：

首先他是以pipeline的方式运行的；pipeline又由多个component组成，现在我们使用的基本pipeline 包括，数据预处理（dataProcess），特征提取（bert featurize），下游分类或者是匹配（classifier or simbert-match）。

能用的component基本都需要自己实现，实现的比较糙，没有他那么精细。

我们实现一个component都会继承他的componnet父类，其中有一些function是必须要实现的：

1.create 用于初始化，加载一些自定义的一些参数

2.train 用于训练当前component的模型，用training_data来传递数据，或者是特征，training_data.training_examples就是所有的训练数据。我们可以把当前component产生的特征写入training_data.training_examples。当然要对每一个example单独写入，不然对应不起来。

```python
example.set("text_features", numpy_features[i])
```

3.process 用于预测，在预测的过程中会用到，message参数用来传递预测用的数据，单条预测，如果我们需要把featurize component里生成的特征传入下一个component，可以用一下方式。

```python
message.set("text_features", numpy_features)
```

然后在下一个component里取出。

4.persist 保存模型，如果你的模型不需要保存，可以不用。

5.load 加载模型，如果模型都没有保存，就不需要加载。

比如在使用bert as service 的过程中我们就不需要保存模型和加载模型，因为我们是每一次都去调用接口的。

**Attention**，在自己写了新的component以后需要去registry.py里面去注册。

参考**simbert**：

首先**import**

```pyhont
from rasa.nlu.classifiers.simbert_match import SimBert_match
```

然后**加入**

```python
component_classes=[....,SimBert_match]
```

即可。

创建服务的时候我们不需要用rasa自带的server，可以用自己写的server。

他自己写的component就是很好的例子，可以模仿着写。

我使用的rasa版本应该是1.5或者1.6版本的，现在已经更新到1.9了，可以看看有什么新的东西。