1. embedding特征的hash方案，现在是LabelEncoder
2. 提供多种embedding方案
    1. 连续特征的处理方案
    2. nn.ModuleDict 还是 一个大的embedding, nn.ModuleDict的话每个field的embedded_size能支持不一样吗，大的embedding怎么解决低频特征过拟合的问题 -> 参考deeprec
3. dataLoader兼容大数据

embedding特征的处理方式：
1. deepctr：LabelEncoder，sklearn的该类没有partial_fit，应对在线更新时需要自己实现，而且sklearn内部fit时会对classes_排序，所以每次更新后映射值会变化，还不如另起炉灶
2. Fuxictr：维护了一个大的词典，但也有个问题，索引会排序，更新了又打乱了

exam:
1. read all data by chunksize  
CPU times: user 5min 18s, sys: 5.38 s, total: 5min 24s
Wall time: 5min 39s  
2. read all data directly  
CPU times: user 5min 21s, sys: 54.8 s, total: 6min 16s
Wall time: 6min 47s  

3. 
```python
with ThreadPoolExecutor(max_workers=20) as executor:
    for d in data:
        executor.submit(self._fit_encoders, d)
```
CPU times: user 3min 20s, sys: 4.36 s, total: 3min 24s
Wall time: 3min 37s  
4. 
```python
with ProcessPoolExecutor(max_workers=20) as executor:
    for d in data:
        executor.submit(self._fit_encoders, d)
```
CPU times: user 2min 37s, sys: 5.78 s, total: 2min 43s
Wall time: 2min 55s  
5. 
```python
for d in data:
    self._fit_encoders(d)
```
CPU times: user 3min 19s, sys: 4.41 s, total: 3min 24s
Wall time: 3min 37s  
6. 
`import dask.dataframe as dd`  
CPU times: user 8min 55s, sys: 46.6 s, total: 9min 42s
Wall time: 2min 52s