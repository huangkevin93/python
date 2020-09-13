import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import re
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC 
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics

# create a feature about crew?

# Survived = Pclass, Sex, Age, [SibSp, Parch] - Family , Fare, Embarked
train = pd.read_csv('/Users/changlonghuang/Documents/Python/Titanic/train.csv')
test = pd.read_csv('/Users/changlonghuang/Documents/Python/Titanic/test.csv')

# get info on the data columns
"""
train.info()
test.info()
print train.describe()
"""

# Cabin is probably correlated with class. passengerID, name, ticket are not important since they are IDs
not_needed_list = ['PassengerId', 'Name', 'Ticket', 'Cabin','SibSp','Parch']
dummy_var_list = ['Sex', 'Embarked', 'Pclass', 'Title']
y = 'Survived'
test_passengers = test['PassengerId']

#==============Function List===============
# fill the missing values in the train and test data
def set_col_to_bool(df, col1, col2, new_col,new_col_2):
    df[new_col] = df[col1] + df[col2]
    df[new_col].loc[df[new_col] > 0] = 1
    df[new_col].loc[df[new_col] == 0] = 0
    df[new_col_2] = df[col1] + df[col2] + 1
    return df

# iterate to drop features in a list and returns dataframe
def drop_list_features(df, list_of_features):
    for feature in list_of_features:
        df = df.drop(feature, axis = 1)
    return df

# apply dummy variables to dataset
def apply_dummy(df,y, dummy_feature_list):
    new_df = pd.get_dummies(df, columns = dummy_feature_list)
    try:
        dummy_x = new_df.drop(y,axis =1)
    except:
        dummy_x = new_df
    return dummy_x

# cross validation of the model
def cross_validation(model, x, y):
    scores = cross_val_score(model, x,y , cv=5)
    print "Accuracy: %.2f (+/-%.2f)" %(scores.mean(), scores.std()*2)
    return scores

# confusion matrix
def confusion_matrix_plot(y_truth, y_pred, model_name):
    cm = metrics.confusion_matrix(y_truth, y_pred)
    ax = plt.axes()
    sns.heatmap(cm, annot= True, fmt = 'd')
    ax.set_title(model_name + " for Titanic Dataset")
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Actual')
    plt.show()

# predict
def predict_on_test(test_df, model):
    new_df = pd.DataFrame()
    new_df['PassengerId'] = test_passengers
    try:
        pred = model.predict(test_df)
    except:
        pred = avg_models(test_df, model)
        pred = pred['avg']
    new_df['Survived'] = pred
    return new_df

# add a column of name length
def name_length_gen(df, col_name, new_col):
    df[new_col] = df[col_name].str.len()
    return df

# get title
def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

def avg_models(data, model_list):
    shape_of_train = (len(data), 1)
    placeholder_total = pd.DataFrame()
    for i in model_list:
        placeholder_total[str(type(i))] = i.predict(data)
    placeholder_total['avg'] = placeholder_total.mean(axis=1).apply(lambda x: int(round(x)))
    print "Averaging models complete..."
    return placeholder_total

# we know that Titanic survivers usually had family
#========Cleaning up the data========
train = set_col_to_bool(train, 'SibSp', 'Parch', 'Family', 'Family_Size')
test = set_col_to_bool(test, 'SibSp', 'Parch','Family','Family_Size')

train = name_length_gen(train, 'Name', 'Length_of_Name')
test = name_length_gen(test, 'Name', 'Length_of_Name')

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

train['Title'] = train["Name"].apply(get_title)
test['Title'] = test["Name"].apply(get_title)

full_data = [train,test]
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train.groupby(['Pclass','Sex','Family'])['Age'].transform(lambda x: x.fillna(x.median()))

test['Age'] = test.groupby(['Pclass','Sex','Family'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Fare'] = test.groupby(['Pclass','Sex','Family'])['Fare'].transform(lambda x: x.fillna(x.median()))

test = drop_list_features(test, not_needed_list)
train = drop_list_features(train, not_needed_list)

test = apply_dummy(test,"", dummy_var_list)
train = apply_dummy(train,"", dummy_var_list)

train_x = train.drop(y,axis = 1)
train_y = train[y]

#==========Modeling===================
#svc model
new_svc = SVC(class_weight='balanced', random_state = 3)
svc_model = new_svc.fit(train_x, train_y)
prediction_1 = svc_model.predict(train_x)
print "Score of SVC is: %.4f" % svc_model.score(train_x,train_y)
#confusion_matrix_plot(train_y, prediction_1, 'SVC')

#logistic regression
log = LogisticRegression(class_weight = 'balanced')
log_model = log.fit(train_x,train_y)
prediction_2 = log_model.predict(train_x)
print "Score of Log is: %.4f" % log_model.score(train_x,train_y)
#confusion_matrix_plot(train_y, prediction_2, 'Log')

#RFC
rfc = RandomForestClassifier(n_estimators = 100, max_features = 9, class_weight='balanced')
rfc_model = rfc.fit(train_x, train_y)
feature_names = list(train_x)
#importances = rfc_model.feature_importances_
#indicies = np.argsort(importances)[::-1]
#prediction_1 = rfc_model.predict(train_x)
print "Score of Random Tree is: %.4f" % rfc_model.score(train_x,train_y)
"""
for j in indicies:
    print "Feature: %s | Importance: %.4f" %(feature_names[j], importances[j])
print cross_validation(rfc_model, train_x,train_y)
print "\n"
confusion_matrix_plot(train_y, prediction_1, 'Random Forest Classifier')
"""

#cart - not as good as random forest
cart = DecisionTreeClassifier(max_features = 6, max_depth=9, class_weight='balanced')
cart_model = cart.fit(train_x, train_y)
feature_names = list(train_x)
importances = cart_model.feature_importances_
indicies = np.argsort(importances)[::-1]
#prediction_2 = cart_model.predict(train_x)
print "Score of CART is: %.4f" % cart_model.score(train_x,train_y)
"""
for j in indicies:
    print "Feature: %s | Importance: %.4f" %(feature_names[j], importances[j])

print cross_validation(cart_model, train_x,train_y)
print "\n"
confusion_matrix_plot(train_y, prediction_2, 'CART Model')
"""

#xgboost
xgb_class = xgb.XGBClassifier(max_depth=3, n_estimators=110, learning_rate=.09,
                              scale_pos_weight = .55)
xgb_model = xgb_class.fit(train_x, train_y)
#xgb.plot_importance(xgb_model)
#plt.show()
prediction_3 = xgb_model.predict(train_x)
#confusion_matrix_plot(train_y, prediction_3, 'XGBoost')
print "Score of XGB is: %.4f" % xgb_model.score(train_x,train_y)
#print cross_validation(xgb_model, train_x,train_y)

#============== Avg Scores ===============
list_of_models = [svc_model, log_model, rfc_model,cart_model, xgb_model]

"""
predictions_df = avg_models(train_x, list_of_models)
confusion_matrix_plot(train_y, predictions_df['avg'], 'avg')
"""
csv_predict = predict_on_test(test, list_of_models)
csv_predict.to_csv(path_or_buf = '/Users/changlonghuang/Documents/Python/Titanic/prediction_avg.csv',index = False)
print "Analysis..complete"