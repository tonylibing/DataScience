from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import processor
reload(processor)
from processor import *

class LogInfoFeature(TransformerMixin):
    #Idx,logTimes,logDays,logDayGap,logGapAVG,mostCountCode,leastCountCode,mostCountCate,leastCountCate,codeEntropy,cateEntropy,logDuration
    def transform(self, df):
        df['logInfo'] = df['LogInfo3'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df['Listinginfo'] = df['Listinginfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d'))
        df['ListingGap'] = df[['logInfo', 'Listinginfo']].apply(lambda x: (x[1] - x[0]).days, axis=1)
        df['LogAction'] = df[['LogInfo1','LogInfo2']].apply(lambda x:'_'.join(x),axis=1)
        action_cnt = df.groupby(['Idx','LogAction'])['LogInfo3'].count().unstack('LogAction').fillna(0)
        action_cnt.columns = ['action_cnt_' + x  for x in action_cnt.columns.values.tolist()]
        action_cnt.reset_index(inplace=True)

        LogTimes = df.groupby(['Idx'])['LogInfo3'].count()
        LogTimes.columns=['logTimes']
        LogTimes.reset_index(inplace=True)

        LogDays = df.groupby(['Idx'])['LogInfo3'].nunique()
        LogDays.columns=['logDays']
        LogDays.reset_index(inplace=True)

        logDayGap =  df.groupby(['Idx'])['LogInfo3'].max()


        return df

    def fit(self, *_):
        return self

class UpdateInfoFeature(TransformerMixin):
    #Idx,ListingInfo1,UserupdateInfo1,UserupdateInfo2
    #Idx, updateTimes, updateDays, updateDayGap, updateGapAVG, mostCountInfo, leastCountInfo, infoEntropy, infoUpdateDuration
    def transform(self, df):
        df['ListingInfo'] = df['ListingInfo1'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
        df['UserupdateInfo'] = df['UserupdateInfo2'].map(lambda x: datetime.datetime.strptime(x, '%Y/%m/%d'))
        data3GroupbyIdx = pd.DataFrame({'Idx': df['Idx'].drop_duplicates()})
        # items = set(df['UserupdateInfo1'].values)
        time_window = [7, 30, 60, 90, 120, 150, 180]
        for tw in time_window:
            df['TruncatedLogInfo'] = df['ListingInfo'].map(lambda x: x + datetime.timedelta(-tw))
            temp = df.loc[df['UserupdateInfo'] >= df['TruncatedLogInfo']]

            # frequency of updating
            freq_stats = temp.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
            data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_freq'] = data3GroupbyIdx['Idx'].map(
                lambda x: freq_stats.get(x, 0))

            # number of updated types
            Idx_UserupdateInfo1 = temp[['Idx', 'UserupdateInfo1']].drop_duplicates()
            uniq_stats = Idx_UserupdateInfo1.groupby(['Idx'])['UserupdateInfo1'].count().to_dict()
            data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_unique'] = data3GroupbyIdx['Idx'].map(
                lambda x: uniq_stats.get(x, 0))

            # average count of each type
            data3GroupbyIdx['UserupdateInfo_' + str(tw) + '_avg_count'] = data3GroupbyIdx[
                ['UserupdateInfo_' + str(tw) + '_freq', 'UserupdateInfo_' + str(tw) + '_unique']]. \
                apply(lambda x: x[0] * 1.0 / x[1], axis=1)

            # Idx_UserupdateInfo1_V2 = Idx_UserupdateInfo1.assign(update_cnt = 1).set_index(["Idx", "UserupdateInfo1"]).unstack("UserupdateInfo1").fillna(0)
            # Idx_UserupdateInfo1_V2.columns = [ x[0]+x[1]+str(tw) for x in Idx_UserupdateInfo1_V2.columns.values.tolist()]
            # Idx_UserupdateInfo1_V2.reset_index(inplace=True)
            Idx_UserupdateInfo1_V2 = temp.groupby(['Idx','UserupdateInfo1'])['UserupdateInfo2'].count().unstack('UserupdateInfo1').fillna(0)
            Idx_UserupdateInfo1_V2.columns =  [ 'update_cnt'+x+'_'+str(tw) for x in Idx_UserupdateInfo1_V2.columns.values.tolist()]
            Idx_UserupdateInfo1_V2.reset_index(inplace=True)

            data3GroupbyIdx=pd.merge(data3GroupbyIdx, Idx_UserupdateInfo1_V2, on='Idx', how='left')

        return data3GroupbyIdx


    def fit(self, *_):
        return self

# classification_model= reload(classification_model)
path= 'D:/workspace/DataScience/data/ppd/'
#path= '/home/tanglek/workspace/DataScience/data/ppd/'
data1 = pd.read_csv(path+'Training Set/PPD_LogInfo_3_1_Training_Set.csv', header = 0)
data2 = pd.read_csv(path+'Training Set/PPD_Training_Master_GBK_3_1_Training_Set.csv', header = 0,encoding = 'gbk')
data3 = pd.read_csv(path+'Training Set/PPD_Userupdate_Info_3_1_Training_Set.csv', header = 0)

def process_data(df):
    data1_pip = Pipeline([('drop_missing',DropColumnTransformer()),
                         ('drop_row',RowMissingDroperTransformer())])

    data2_pip = Pipeline([('lower_string', StripTransformer('UserInfo_9'))])

    data3_pip = Pipeline([('lower_string',LowerTransformer('UserupdateInfo1')),
                          ('strip_city',StripTransformer(['UserInfo_9'],'市'))])

    d1=data1_pip.fit(data1)
    d2 = data2_pip.fit(data2)
    d3=data3_pip.fit(data3)





#features with date, category manipulation without feature importance
def get_basic_features(df, ordinal, categorical,date_manip ,cont):
    date_pip = Pipeline([('extract',column_extractor.column_extractor(date_manip)),
                ('date_manip', date_transformer.date_transformer('%Y-%m-%d')),
                ('d-m-y-q-dow', FeatureUnion([('day',date_transformer.day_of_month_transformer()),
                                            ('month',date_transformer.month_transformer()),
                                            ('dow',date_transformer.day_of_week_transformer()),
                                            ('quarter',date_transformer.month_quarter_transformer()),
                                            ('year',date_transformer.year_transformer())])),
                ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    continuous = Pipeline([
            ('extract', column_extractor.column_extractor(cont)),
            ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent'))),
            ('scale', Normalizer())])

    ordinal_pip = Pipeline([('extract', column_extractor.column_extractor(ordinal)),
                            ('ord', categorical_transformer.ordinal_transformer(ordinal)),
                           ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    one_hot = Pipeline([('extract', column_extractor.column_extractor(categorical)),
                        ('lab_enc', categorical_transformer.label_transformer()),
                        ('one_hot', ModelTransformer.ModelTransformer(OneHotEncoder(sparse=False)))])


    features = Pipeline([('parallel', FeatureUnion([('date',date_pip),
                                                    ('continuous',continuous),
                                                    ('ordinal_pip',ordinal_pip),
                                                    ('one_hot',one_hot)])),
                        ('cleanup',cleanup_transformer.cleanup_transformer())])

    return features, features.transform(df)

#features with date, category manipulation without feature importance
def get_poly_features(df, ordinal, categorical,date_manip ,cont):
    date_pip = Pipeline([('extract',column_extractor.column_extractor(date_manip)),
                ('date_manip', date_transformer.date_transformer('%Y-%m-%d')),
                ('d-m-y-q-dow', FeatureUnion([('day',date_transformer.day_of_month_transformer()),
                                            ('month',date_transformer.month_transformer()),
                                            ('dow',date_transformer.day_of_week_transformer()),
                                            ('quarter',date_transformer.month_quarter_transformer()),
                                            ('year',date_transformer.year_transformer())])),
                ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    continuous = Pipeline([
            ('extract', column_extractor.column_extractor(cont)),
            ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent'))),
            ('scale', Normalizer())])

    ordinal_pip = Pipeline([('extract', column_extractor.column_extractor(ordinal)),
                            ('ord', categorical_transformer.ordinal_transformer(ordinal)),
                           ('impute',imputing_transformer.imputing_transformer(Imputer(strategy='most_frequent')))])

    one_hot = Pipeline([('extract', column_extractor.column_extractor(categorical)),
                        ('lab_enc', categorical_transformer.label_transformer()),
                        ('one_hot', ModelTransformer.ModelTransformer(OneHotEncoder(sparse=False)))])


    features = Pipeline([('parallel', FeatureUnion([('date',date_pip),
                                                    ('continuous',continuous),
                                                    ('ordinal_pip',ordinal_pip),
                                                    ('one_hot',one_hot)])),
                        ('poly',ModelTransformer.ModelTransformer(PolynomialFeatures(degree=2))),
                        ('cleanup',cleanup_transformer.cleanup_transformer())])

    return features, features.transform(df)
