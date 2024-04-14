import numpy as np
#from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split,GridSearchCV
from matplotlib import pyplot as plt
#from matplotlib.colors import ListedColormap
#from sklearn.metrics import confusion_matrix 
#import seaborn as sns
#import pandas as pd


def plot_decision_boundary(X, y, X1, X2,classifier, plot_mean=True):
    X_set, y_set= X, y
    plt.figure(figsize=(10, 10))
    cmap=plt.cm.coolwarm
    plt.contourf(X1, X2, classifier, cmap=cmap, alpha=0.9)
    plt.scatter(X_set[:, 0], X_set[:, 1], c=y_set, cmap=cmap, marker='o')
    if plot_mean:
        mean_0 = np.mean(X_set[y_set == 0], axis=0)
        mean_1 = np.mean(X_set[y_set == 1], axis=0)
        plt.scatter(mean_0[0], mean_0[1], c='black', cmap=cmap, marker='x')
        plt.scatter(mean_1[0], mean_1[1], c='green', cmap=cmap, marker='x')
    
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    plt.title('SVM DECISION BOUNDARY')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    return plt.show()

def make_meshgrid(X, y, step):
      #Make a meshgrid covering the range of X. This is used to display classification regions.
      X_set, y_set= X, y
      x_start, x_stop = X_set[:, 0].min() - 1, X_set[:, 0].max() + 1
      y_start, y_stop = X_set[:, 1].min() - 1, X_set[:, 1].max() + 1
      X1,X2=np.meshgrid(np.arange(x_start, x_stop, step), np.arange(y_start, y_stop, step))
      return X1,X2
