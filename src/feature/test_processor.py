from processor import OutliersFilter

import numpy
import pandas
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from itertools import combinations


def load_problem_flight(large=False, convert_to_ints=False):
    '''
    Dataset used in common ML benchmarks: https://github.com/szilard/benchm-ml
    links to files:
    https://s3.amazonaws.com/benchm-ml--main/test.csv
    https://s3.amazonaws.com/benchm-ml--main/train-0.1m.csv
    https://s3.amazonaws.com/benchm-ml--main/train-1m.csv
    https://s3.amazonaws.com/benchm-ml--main/train-10m.csv
    '''
    if large:
        trainX = pandas.read_csv('../data/flight_train-10m.csv')
    else:
        trainX = pandas.read_csv('../data/flight_train-0.1m.csv')
    testX = pandas.read_csv('../data/flight_test.csv')

    trainY = (trainX.dep_delayed_15min.values == 'Y') * 1
    testY = (testX.dep_delayed_15min.values == 'Y') * 1

    trainX = trainX.drop('dep_delayed_15min', axis=1)
    testX = testX.drop('dep_delayed_15min', axis=1)
    if convert_to_ints:
        categoricals = ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier', 'Origin', 'Dest', ]
        continous = ['Distance']

        trainX, testX = process_categorical_features(trainX, testX, columns=categoricals)
        trainX, testX = process_continuous_features(trainX, testX, columns=continous)

        trainX['DepTime'] = trainX['DepTime'] // 100
        testX['DepTime'] = testX['DepTime'] // 100

    return trainX, testX, trainY, testY