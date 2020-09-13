import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics

zero_features = ['MasVnrArea']
none_features = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond',
                'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                'FireplaceQu', 'GarageType', 'GarageYrBlt',
                'GarageFinish', 'GarageQual', 'GarageCond',
                'PoolQC', 'Fence', 'MiscFeature']
drop_row = ['Electrical']


def main():
    train = pd.read_csv('/Users/changlonghuang/Documents/Python/Housing Prices/train.csv')
    test = pd.read_csv('/Users/changlonghuang/Documents/Python/Housing Prices/test.csv')

    list_of_df = [train, test]

    print 'train'
    print train.head()
    print train.info()
    print train.describe()

    print 'test'
    print test.head()
    print test.info()
    print test.describe()

if __name__ == '__main__':
    main()