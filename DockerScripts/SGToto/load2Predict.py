import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from functools import reduce

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn import utils, preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn import model_selection
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from sklearn.linear_model import RidgeClassifier, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.svm import SVC, SVR

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#from skmultilearn.adapt import MLkNN
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
#from sklearn.linear_model import RidgeRegressor
from sklearn.ensemble import GradientBoostingRegressor
# Compare Algorithms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import statsmodels.api as sm
from sklearn.metrics import accuracy_score 


print(os.listdir("../input"))

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
#warnings.filterwarnings("ignore") #Hide messy Numpy warnings
#warnings.filterwarnings('ignore', category=DeprecationWarning)


seed = 21

def getAccuracy(prediction, actual):
    matched = []
    for (p,a) in zip(prediction, actual):
        matched.append(len(set(p.astype(int)) & set(a)))
    return matched

def getPercent(prediction, actual, n):
    matched = []
    for (p,a) in zip(prediction, actual):
        matched.append(set(p.astype(int)) & set(a))
    return sum(len(num) > n for num in matched)/len(matched)*100.00

def getCounts(prediction, actual):
    matched = []
    N = [0,0,0,0,0,0,0,0]
    if len(prediction) == 0: return N
    for (a,p) in zip(actual, prediction):
        matched.append(set(p.astype(int)) & set(a))
    if len(matched) == 0:
        return N
    N[0] = sum(len(num) > 0 for num in matched)/len(matched)*100.00
    N[1] = sum(len(num) > 1 for num in matched)/len(matched)*100.00
    N[2] = sum(len(num) > 2 for num in matched)/len(matched)*100.00
    N[3] = sum(len(num) > 3 for num in matched)/len(matched)*100.00
    N[4] = sum(len(num) > 4 for num in matched)/len(matched)*100.00
    N[5] = sum(len(num) > 5 for num in matched)/len(matched)*100.00
    N[6] = sum(len(num) > 6 for num in matched)/len(matched)*100.00
    N[7] = sum(len(num) > 7 for num in matched)/len(matched)*100.00
#    matched.extend(N)
    return N


def getAccuracy1dCount(prediction, actual):
    iMatched = 0
    print(len(prediction))
    for i in range(0,len(prediction)):
        if prediction[i] == actual[i]:
            iMatched = iMatched +1
    return iMatched

def getAccuracy1dPercentCorrect(prediction, actual):
    iMatched = 0
    print(len(prediction))
    for i in range(0,len(prediction)):
        if prediction[i] == actual[i]:
            iMatched = iMatched +1
    return iMatched/len(prediction) * 100.0


def getAccuracyCount(prediction, actual):
    matched = []
    if len(prediction) == 0: return 0
    for (p,a) in zip(prediction, actual):
#        print ( "p: ", p, " a: ", a, (set(p.astype(int)) & set(a)) )
        matched.append((set(p.astype(int)) & set(a)))
    return sum(len(num) > 0 for num in matched)/len(matched)*100.00

def getIntersection(p1, p2):
    return [reduce(np.intersect1d, (p.astype(int), a.astype(int))) for (p,a) in zip(p1, p2)]

def getUnion(p1, p2):
    if len(p1) == 0: return p2
    return [reduce(np.union1d, (p.astype(int), a.astype(int))) for (p,a) in zip(p1, p2)]

def getUnion1dArray(p1,p2):
    return reduce(np.union1d, (p1,p2))

def getIntersection1dArray(prediction, actual):
    iMatched = 0
#     return reduce(np.intersect1d,([prediction,actual]))
#     for idx, i in enumerate(prediction):
#       if prediction[idx] == actual[idx]:
#         iMatched = iMatched + 1
#     return iMatched

# def getAccuracy1dCount(prediction, actual):
#     iMatched = 0
#     for idx in enumerate(prediction):
#       if prediction[idx] == actual[idx]:
#         iMatched = iMatched + 1
#     return iMatched



def unionPrediction(ff):
    predicted = []
    for i, f in enumerate(ff):
#        i_index = name_.index(sInputDir + f + '.csv')        
        i_index = name_.index(f)        
        if i == 0: predicted = list_[i_index][cols].values
        if i > 0:
            predicted = getUnion(predicted,list_[i_index][cols].values)
    return predicted

def getMatches(prediction, actual):
    matched = []
    for (p,a) in zip(prediction, actual):
        print ( len(p), "** p: ", p, " a: ", a, (set(p.astype(int)) & set(a)) )
        matched.append((set(p.astype(int)) & set(a)))
    return matched

def getMatchedCount(prediction, actual):
    return getAccuracyCount(prediction, actual)

def printPredictions ( prediction, actual ):
    for (p,a) in zip(prediction, actual):
        print ( '[',len(set(p.astype(int) )&set(a.astype(int))),'/',len(p),'',set(p.astype(int) ), ' ', set(a.astype(int)), ' ', set(p.astype(int) )&set(a.astype(int)))

def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[0], bins[-1])

def getColums (idx):
    return list(os.path.splitext(basename(name_[idx]))[0][2:])


def showResult(str_alg, prediction, actual ):
    print( str_alg, " Accuracy predict 1 in 7: ", getAccuracy(prediction, actual))

def printResult(predictions, actual):
    df_result=pd.DataFrame({ 'Predicted':list(predictions), 'Actual':list(actual)})
    print(df_result)  
    
def getStdDeviationOfPrediction(ytestPredicted, y_test):
    print("Removing invalid predictions ( < 0 )")
    ind2remove = np.where(ytestPredicted <= 0)[0]
    ytestPredictedFinal = np.delete(ytestPredicted, ind2remove)
    y_testFinal = np.delete(y_test, ind2remove)
    iM = getAccuracy1dCount(y_testFinal, ytestPredictedFinal)
    print(iM)   
    print(np.std(ytestPredictedFinal-y_testFinal))


print('Done')

def loadResult(lresult, col_n = 1):
#    col_n = 1  #Column Number interested
#    print(lresult)
    aa = np.delete(lresult, np.s_[col_n:], axis=1)  
    aa = np.delete(aa, np.s_[0:col_n-1], axis=1)  

    # 1. INSTANTIATE
    enc = preprocessing.OneHotEncoder()

    # 2. FIT
    enc.fit(aa)

    # 3. Transform
    onehotlabels = enc.transform(aa).toarray()
    onehotlabels.shape
    #print(onehotlabels)

    #Convert 2d array to Dataframe
    y = pd.DataFrame(aa, columns=list('N'))
    y.head()
    y = aa.astype(int).ravel()
#    print ( y )
    return y

def loadTotoData():
    pp = pd.read_csv('../input/PPv3.csv')
#    print(pp.tail())

    lr = pd.read_csv('../input/SGH.csv')
    #print(lr.describe())

    #print(lr)

    #print(len(lr))
    #lr = lr.sort_values(by=['D'])
    #lr = lr.drop_duplicates () ;
    print(len(lr))
    cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
    lr = lr[cols]
    #print(lr.head(30))

    #https://pandas.pydata.org/pandas-docs/stable/merging.html
    df = pd.concat([pp, lr], axis=1, sort=False)
    df = df.dropna()
    #print(len(df))
    #df.head()
    df.reset_index().drop(['D'], axis=1)

    cols = ['N1','N2','N3','N4','N5','N6','N7']
    lr = df[cols]


    # cols = ['L','M','S','R','E','A','V' ,'J','U']
    # cols = ['L', 'M', 'R', 'J', 'U']
    # X = df[cols]

    X = df
    #Remove T, D and Results
    drop_cols = ['T','D','N1','N2','N3','N4','N5','N6','N7','L','M','S','R','E','A','V' ,'J','U','K']
    X = X.drop(drop_cols, axis=1)
    related_X = X
    dataset = related_X

    lresult = np.sort(lr.values[:, ::-1])
#    print(lresult)

#    loadResult(lresult, 1) # interested in column 1 
    return lresult, dataset 

