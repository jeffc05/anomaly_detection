import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


def data_generator(contamination=0.05, n_train=500, n_test=500, n_features=6, random_state=123):
    # contamination: percentage of outliers
    # n_train: number of training points
    # n_test: number of testing points
    # n_features: number of features
    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train,
        n_test=n_test,
        n_features= n_features,
        contamination=contamination,
        random_state=random_state)
    
    X_train = 5 - X_train
    X_test = 5 - X_test
    X_train_pd = pd.DataFrame(X_train)
    X_test_pd = pd.DataFrame(X_test)
    
    return X_train_pd, X_test_pd, y_train, y_test

def plot_data(X, y):
    plt.scatter(X[0], X[1], c=y, alpha=0.8)
    plt.title('Scatter plot')
    plt.xlabel(f'x0')
    plt.ylabel(f'x1')
    plt.show()

def count_stat(vector):
    # Because it is '0' and '1', we can run a count statistic.
    unique, counts = np.unique(vector, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

def descriptive_stat_threshold(df, feature_list, pred_score, threshold):
    # Let's see how many '0's and '1's.
    df = pd.DataFrame(df)
    df.columns = feature_list
    df['Anomaly_Score'] = pred_score
    df['Group'] = np.where(df['Anomaly_Score'] < threshold, 'Normal', 'Outlier')

    # Now let's show the summary statistics:
    cnt = df.groupby('Group')['Anomaly_Score'].count().reset_index().rename(columns={'Anomaly_Score': 'Count'})
    cnt['Count %'] = (cnt['Count'] / cnt['Count'].sum()) * 100 # The count and count %
    stat = df.groupby('Group').mean().round(2).reset_index() # The avg.
    stat = cnt.merge(stat, on='Group') # Put the count and the avg. together
    return stat

def confusion_matrix1(actual, score, threshold):
    actual_pred = pd.DataFrame({'Actual': actual, 'Pred': score})
    actual_pred['Pred'] = np.where(actual_pred['Pred'] <= threshold, 0, 1)
    cm = pd.crosstab(actual_pred['Actual'], actual_pred['Pred'])
    return cm

def confusion_matrix2(actual, pred):
    actual_pred = pd.DataFrame({'Actual': actual, 'Pred': pred})
    cm = pd.crosstab(actual_pred['Actual'], actual_pred['Pred'])
    return cm

def evaluate_outlier_classifier(model, data):
    # Get labels
    model.fit(data)
    labels = model.labels_
    # Return inliers
    return data[labels == 0]

def evaluate_regressor(inliers):
    X = inliers.drop("price", axis=1)
    y = inliers[['price']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10)

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    preds = lr.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)

    return round(rmse, 3)