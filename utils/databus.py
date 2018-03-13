import numpy as np
import pandas as pd
import os
import sys
import argparse
import pickle
from sklearn.datasets import dump_svmlight_file
from datetime import datetime
from datetime import timedelta
from dateutil.parser import parse
import xgboost
import lightgbm
import tensorflow as tf

from xgboost import XGBClassifier
from collections import Counter



class DataBus()
    def __init__(self,platform):
        self.platform = platform

    def read_csv(self,path):
        if self.platform == 'cloud':
            with tf.gfile.FastGFile(path, 'rb') as gf:
                data = pd.read_csv(gf)
        elif self.platform == 'standalone':
            data = pd.read_csv(path,'rb')

        return data

    def read_csv(self, path):
        if self.platform == 'cloud':
            with tf.gfile.FastGFile(path, 'rb') as gf:
                data = pd.read_csv(gf)
        elif self.platform == 'standalone':
            data = pd.read_csv(path, 'rb')

        return data

    def load_pickle(self):

