import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import graphviz
import math
import seaborn as sns
get_ipython().magic('matplotlib inline')

#Referenced and modified Rayid's magic loop
#Magic loop https://github.com/rayidghani/magicloops
#Used stackoverflow for guidance on visualizations and dataframe operations


'''
Reading in data as pandas dataframe from filepath and exension.e e
'''
def read_data(filepath, file_type = None):
    if file_type == 'csv':
        full_df = pd.read_csv(filepath)
    elif file_type == 'excel':
        full_df = pd.read_excel(filepath)
    elif file_type == 'json':
        full_df = pd.read_json(filepath)
    return full_df

'''
Filtering dataframe column by equality
'''
def filter_df_by_equality(df, var, value):
    df = df[(df[var] == value)]
    return df

'''
Filtering dataframe column by start and end dates.
'''
def filter_df_by_date_range(df, date_var, start, end):
    df = df[(df[date_var] >= start) & (df[date_var] <= end)]
    return df

'''
Function for adding dummy variable.
'''
def add_dummy_variable(df, var, dummy_var, lambda_equation):
    df[dummy_var] = df[var].apply(lambda_equation)

'''
Train/test split of the data for use in machine learning models.
'''
def train_test_data_split(df, var, test_size):
    X = df
    Y = df[var]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test

'''
Pre-process and describe the data.
'''
def pre_process(df):
    return df.describe()

'''
Removes outliers (above 99th percentile of field) from dataframe
'''
def remove_outliers(df, var):
    quant = df[var].quantile(0.99)
    df[df[var] < quant]
    return df

'''
Converts t/f values in dataframe columnns to 1's and 0's
'''
def convert_true_false_1_0(df):
    col_list = list(df.columns)
    for col in col_list:
        if 'f' in list(df[col].unique()):
            df[col] = df[col].apply(lambda x: 1 if x=='t' else 0)

'''
Returns a list of binary categorical variables in a dataframe
'''
def binary_categorical_vars(df):
    binary_categorical_vars_list = []
    for column in df.columns:
        if len(df[column].unique()) <= 2:
            binary_categorical_vars_list.append(column)
    return binary_categorical_vars_list

'''
Returns a list of non-binary categorical variables in a dataframe
'''
def non_binary_categorical_vars(df, binary_categorical_vars_list):
    non_binary_categorical_vars_list = []
    for column in df.columns:
        if column not in binary_categorical_vars_list:
            non_binary_categorical_vars_list.append(column)
    return non_binary_categorical_vars_list

'''
Creates dummy variables for non-binary categorical variables, joins to
original dataframe, and drops variables from which dummies were created.
'''
def join_and_drop_dummies(df_to_join_to, vars_list):
    for each in vars_list:
        dummies_df = pd.get_dummies(df_to_join_to[each])
        df_to_join_to = df_to_join_to.join(dummies_df)
        df_to_join_to = df_to_join_to.drop(each, axis = 1)
    return df_to_join_to

'''
Simply creates dummies from categorical variables
'''
def get_dummies_from_categorical_var(df, var_list):
    dummies_df = pd.get_dummies(df[var_list])
    return dummies_df

'''
Joins created dummy variables to original dataframe
'''
def join_dummies(df_to_join_to, dummies_df):
    df_to_join_to = df_to_join_to.join(dummies_df)
    return df_to_join_to

'''
Drops drops variables from which dummies were created in original dataframe
'''
def drop_non_dummy_cat_var(df, var_list):
    for var in var_list:
        df = df.drop(var, axis = 1)
    return df

'''
Temporal train test split function for machine learning models based on
different start and end dates for test/training sets
'''
def temporal_train_test_data_split(df, features, filter_var, outcome_var,  start_date):
    if start_date == '2011-01-01':
        training_end_date = '2011-12-31'
        testing_end_date = '2012-06-31'

    elif start_date == '2011-07-01':
        training_end_date = '2012-07-31'
        testing_end_date = '2012-12-31'

    elif start_date == '2012-01-01':
        training_end_date = '2012-12-31'
        testing_end_date = '2013-06-31'

    y_df = df[[filter_var, outcome_var]]

    x_train_filter_by_date = filter_df_by_date_range(df, filter_var, training_end_date, testing_end_date)
    x_test_filter_by_date = filter_df_by_date_range(df, filter_var, start_date, training_end_date)

    y_train_filter_by_date = filter_df_by_date_range(y_df, filter_var, training_end_date, testing_end_date)
    y_test_filter_by_date = filter_df_by_date_range(y_df, filter_var, start_date, training_end_date)

    x_train = x_train_filter_by_date[features]
    x_test = x_test_filter_by_date[features]

    y_train = y_train_filter_by_date[outcome_var]
    y_test = y_test_filter_by_date[outcome_var]

    return x_test, x_train, y_test, y_train

'''
Sums number of missing values per variable in a dataframe
'''
def find_missing_values(df):
    missing_df = df.isnull().sum().sort_values(ascending = False)
    return missing_df

'''
Imputes missing values with mean, median, or mode of other values
in a dataframe field
'''
def impute_missing_values(df, var, fill_method = None):
    if fill_method == 'mean':
        df[var] = df[var].fillna(df[var].mean())
    if fill_method == 'mode':
        df[var] = df[var].fillna(df[var].mode())
    if fill_method == 'median':
        df[var] = df[var].fillna(df[var].median())

'''
Visualizes a correlation heatmap from a dataframe'''
def visualize_correlation_heatmap(df):
    # corr = df.corr()
    # sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    # plt.show()
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)

'''
Visualizes a correlation heatmap table from dataframe
'''
def magnify(df):
    corr = df.corr()
    sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
    cmap = cmap=sns.diverging_palette(5, 250, as_cmap=True)
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]

    corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())

'''
Visualizes line graph from dataframe and column/s
'''
def visualize_line_graph(df, var, title, xlabel, ylabel):
    x = list(df.index.values)
    y = list(df[var].values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x,y)
    plt.show()

'''
Visualizes bar chart from dataframe and column/s
'''
def visualize_bar_chart(xvals, yvals, xlabel, ylabel, title, width = 0.35, color = 'blue'):
    counts = yvals

    ind = np.arange(len(xvals))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, counts, width, color=color)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(xvals)

    plt.gcf().subplots_adjust(bottom=0.2)

    plt.show()

'''
Instantiating machine learning models
'''

def LR():
    return LogisticRegression(penalty = 'l1', C = 1e5)

def KNN():
    return KNeighborsClassifier(n_neighbors = 3)

def DT():
    return DecisionTreeClassifier()

def SVM():
    return svm.SVC(kernel = 'linear', probability = True, random_state = 3)

def RF():
    return RandomForestClassifier(n_estimators = 50, n_jobs = -1)

def AB():
    return AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                                                    algorithm="SAMME",
                                                    n_estimators=200)

def GB():
    return GradientBoostingClassifier(learning_rate = 0.05,
                                    	subsample = 0.5,
                                    	max_depth = 6,
                                    	n_estimators = 10)
def NB():
    return GaussianNB()
'''
'''

'''
Returns classifier accuracy score from train/test split data
'''
def classifier_accuracy_score(classifiers_list, X_train, Y_train, X_output, Y_output):
    classifier_and_score_list = []
    for classifier in classifiers_list:
        accuracy_score = classifier.score(X_output, Y_output)
        classifier_and_score_list.append((classifier, accuracy_score))
    return classifier_and_score_list

'''
Returns recall threshold from train/test split data
'''
def precision_recall_threshold(y_test, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    return precision, recall, thresholds

'''
Plots precision recall curves from train/test split data
'''
def plot_precision_recall_curve(model_instance, x_test, x_train, y_test, y_train):
    random_state = np.random.RandomState(0)
    classifier = model_instance
    classifier.fit(x_train, y_train)
    y_score = classifier.decision_function(x_test)

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt

    precision, recall, _ = precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
plt.title(model_instance)
