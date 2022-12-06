'''
@File         :features.py
@Description  :featuremap namedtuple
@Time         :2022/12/06 10:10:33
@Author       :tangs
@Version      :1.0
'''

from collections import namedtuple, OrderedDict


class FeatureMap(
    namedtuple(
        'FeatureMap',
        [
            'num_fields',
            'num_features',
            'sparse_features',
            'dense_features',
            'input_length',
            'features_attr',
        ],
    )
):
    """A namedtuple that represents a featuremap.
    It has 4 fields:
    num_fields - nums of raw features
    num_features - nums of vocab size
    input_length - nums of input length
    features_attr - feature attributes
    """

    __slots__ = ()

    def __new__(
        cls,
        num_fields=0,
        num_features=0,
        sparse_features=0,
        dense_features=0,
        input_length=0,
        features_attr=OrderedDict(),
    ):
        return super().__new__(
            cls,
            num_fields,
            num_features,
            sparse_features,
            dense_features,
            input_length,
            features_attr,
        )
