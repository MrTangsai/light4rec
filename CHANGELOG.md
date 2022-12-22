# CHANGELOG

开发日志纪要.


## 2022-12-02

---

基于lightning，模型主要继承自`BaseModel(pl.LightningModule)`，该基类实现了基本的数据清洗转换、特征映射和模型训练推理基本逻辑.
> 这里主要参考了deepctr-torch,该库是将连续特征和类别特征分开处理的，没有形成统一的embedding，这样会导致每个具体的算法forward时会更复杂，预处理方式是直接MinMaxScaler和LabelEncoder


## 2022-12-05

---

利用yaml文件存储基本的数据逻辑，初步实现连续特征和分类特征的处理


## 2022-12-06

---

模型层和数据层分开，各管各的，中间的通信通过featuremap，featuremap是基于一开始的yaml文件去拟合数据得到的大致数据分布，喂给模型层用于初始化embedding的网络结构，看的FuxiCTR写的  
数据类主要就靠 `BaseRecData` 类


## 2022-12-14

---

1. featuremap 得到的是各个特征的size和转换成模型数据的路径，和实际转换成模型数据的函数分开
2. 增加一个装饰器函数，打运行时间日志的，用于比较哪种方法得到featuremap更快，尝试过pandas的chunksize迭代、`dask.dataframe` 和 `concurrent.futures`，最后还是选择了对pandas迭代着来处理特征，因为目前不管是multiprocessing还是concurrent，都不支持迭代器，那如果是超出内存的数据量，我不可能通过 `np.split_array` 或者其他构造list的方式来得到多个数据集合去做并行，dask其实效果还不错，但最后试验还是chunksize快点，用dask应该还可以优化

## 2022-12-22

---
1. 考虑大数据量下数据的迭代方式，目前主流是hdf5的迭代加载方式，截至目前torchdata的`IterDataset`写着还可以，就顺道一起实现了，目前的实现逻辑就是初次加载是会直接形成可迭代的内存数据，同时序列化成npy，后续加载就直接在npy上加载了
2. 考虑加速源数据映射成可用数据的函数，`concurrent.futures`
3. 去掉对pandas的依赖，所有数据处理方式都依靠torch或者numpy，源数据的读取依靠torchdata的datapipe，新有的torcharrow有对dataframe的支持，但感觉还需要发展一段时间，去掉dataframe主要也是想节约点内存