'''
@File         :encoders.py
@Description  :
@Time         :2022/12/08 16:26:14
@Author       :tangs
@Version      :1.0
'''

import numpy as np
from collections import Counter


class Tokenizer:
    def __init__(
        self,
        min_freq=1,
        na_value=None,
        oov_token=0,
        padding=False,
        splitter=None,
        max_len=None,
        share_embedding=None,
    ) -> None:
        super().__init__()
        self._min_freq = min_freq
        self._na_value = na_value
        self.oov_token = oov_token
        self.use_padding = padding
        self.splitter = splitter
        self.max_len = max_len
        self.share_embedding = share_embedding
        self.word_counts = Counter()

    def fit(self, y):
        self.word_counts.update(y.tolist())

    def partial_fit(self, y):
        if self.splitter:
            self.word_counts.update(self.splitter.join(y).split(self.splitter))
        else:
            self.word_counts.update(y.tolist())

    def build_vocab(self):
        word_counts = sorted(self.word_counts.items(), key=lambda x: (-x[1], x[0]))
        words = []
        for token, count in word_counts:
            if count >= self._min_freq:
                if self._na_value is None or token != self._na_value:
                    words.append(token)
        self.vocab = dict(
            (token, idx) for idx, token in enumerate(words, 1 + self.oov_token)
        )
        self.vocab["__OOV__"] = self.oov_token
        if self.use_padding:
            self.vocab["__PAD__"] = (
                len(words) + self.oov_token + 1
            )  # use the last index for __PAD__
        self.vocab_size = len(self.vocab) + self.oov_token

    def encode_category(self, categories):
        category_indices = [[self.vocab.get(x, self.oov_token)] for x in categories]
        return np.array(category_indices)

    def encode_sequence(self, texts):
        sequence_list = [
            [self.vocab.get(x, self.oov_token) for x in text.split(self.splitter)]
            for text in texts
        ]
        sequence_list = [
            sequence[: self.max_len]
            if len(sequence) >= self.max_len
            else sequence + (self.max_len - len(sequence)) * [self.vocab_size - 1]
            for sequence in sequence_list
        ]
        return np.array(sequence_list)
