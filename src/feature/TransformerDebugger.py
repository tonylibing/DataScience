from sklearn.base import TransformerMixin
import pandas as pd
import random


class TransformerDebugger(TransformerMixin):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        self.transformer.fit(X, y)
        return self

    def transform(self, X):
        print(self.transformer.__class__.__name__)
        idx = random.randrange(0, len(X))
        print("Before", "=" * 40)
        print(X[idx])
        X = self.transformer.transform(X)
        print("After ", "=" * 40)
        print(X[idx])
        return X