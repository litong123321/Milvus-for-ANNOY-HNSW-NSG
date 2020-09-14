import numpy as np
from milvus import Milvus, IndexType, MetricType
import time

t1 = time.time()
# 初始化一个Milvus类，以后所有的操作都是通过milvus来的
milvus = Milvus(host='localhost', port='19530')
vec_dim = 1000 #向量的维度
num_vec=100 #向量数量10000，后面还加入了一条和查询向量一样的做验证
#删除collection
milvus.drop_collection('test01')
#创建collection
param = {'collection_name':'test01', 'dimension':vec_dim, 'index_file_size':1024, 'metric_type':MetricType.L2}
milvus.create_collection(param)
#建立分区
milvus.create_partition('test01', 'tag01')

#nlist聚类为多少簇
ivf_param = {'nlist': 16}
milvus.create_index('test01', IndexType.IVF_FLAT, ivf_param) #可以不写这个，默认方式IndexType.FLAT
# 随机生成一批向量数据
vectors_array = np.random.rand(num_vec,vec_dim)
vectors_list = vectors_array.tolist()
vectors_list.append([1 for _ in range(vec_dim)])
ids_list = [i for i in range(len(vectors_list))]
#单次插入的数据量不能大于 256 MB，插入后存在缓存区，缓存区大小由参数index_file_size决定，默认1024M
milvus.insert(collection_name='test01', records=vectors_list, partition_tag="tag01",ids=ids_list)
#一些信息
print(milvus.list_collections())
print(milvus.get_collection_info('test01'))
print(milvus.get_index_info('test01'))
t2 = time.time()
print("create cost:",t2-t1)
# # 创建查询向量
query_vec_list = [[1 for _ in range(vec_dim)]]
# 进行查询, 注意这里的参数nprobe和建立索引时的参数nlist 会因为索引类型不同而影响到查询性能和查询准确率
#IVF_FLAT下查询多少个簇，不能超过nlist
search_param = {'nprobe': 10}
#现在数据还在内存，需要数据落盘，保存到数据库中去，不然查不到数据
milvus.flush(collection_name_array=['test01'])
results = milvus.search(collection_name='test01', query_records=query_vec_list, top_k=10,params=search_param)
print(results)
print("search cost:",time.time()-t2)
#
# sudo docker run -d --name milvus_cpu \
# -p 19530:19530 \
# -p 19121:19121 \
# -v /home/$USER/milvus/db:/var/lib/milvus/db \
# -v /home/$USER/milvus/conf:/var/lib/milvus/conf \
# -v /home/$USER/milvus/logs:/var/lib/milvus/logs \
# -v /home/$USER/milvus/wal:/var/lib/milvus/wal \
# milvusdb/milvus:0.10.2-cpu-d081520-8a2393
