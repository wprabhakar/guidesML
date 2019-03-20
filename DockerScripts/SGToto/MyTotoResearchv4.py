import pandas as pd
import numpy as np
from sklearn import utils, preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from matplotlib.pyplot import figure
from functools import reduce
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import operator



class MyTotoResearch:
    
    @classmethod
    def __init__(self, algo_no=0, inputPPFile='../input/PPv4.csv', inputTotoResult='../input/SGH.csv'):
        self.algo_number = algo_no
        print('Loaded MyTotoResearch algo_no: ', self.algo_number)

    @classmethod
    def load_totodata(self, inputPPFile='../input/PPv4.csv', inputTotoResult='../input/SGH.csv'):
        pp = pd.read_csv(inputPPFile)
        lr = pd.read_csv(inputTotoResult)
        print(len(lr))
        cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
        #lr = lr[cols]

        result_sorted = np.sort(lr[cols].values, axis=1).astype(int)
        result_df = pd.DataFrame(result_sorted, columns=['N1','N2','N3','N4','N5','N6','N7','D'])

        lr = result_df
        print(lr)

        #https://pandas.pydata.org/pandas-docs/stable/merging.html
        df = pd.concat([pp, lr], axis=1, sort=False)
        df = df.dropna()
        df.reset_index().drop(['D'], axis=1)

#         print(df.shape)
#         counts = [df['N'+str(i)].value_counts() for i in range(1,8)]
#         for i in range(len(counts)):
#             df = df[~df['N'+str(i+1)].isin(counts[i][counts[i] <= 3].index)]
#         print('After removing numbers that have shown 3 or less times')
#         print(df.shape)

        cols = ['N1','N2','N3','N4','N5','N6','N7']
        lr = df[cols]

        self.df = df
        self.lresult = np.sort(lr.values[:, ::-1])
        return self.lresult, self.df 
    
    @classmethod
    def modified_dataset ( self, dataset ):
        self.dataset = dataset
        return self.dataset
    
    @classmethod
    def get_result_n(self, col_n):
        aa = np.delete(self.lresult, np.s_[col_n:], axis=1)  
        aa = np.delete(aa, np.s_[0:col_n-1], axis=1)  
        return pd.DataFrame(aa, columns=list('N'))

    @classmethod
    def get_result_n_encoded(self, col_n):
        aa = np.delete(self.lresult, np.s_[col_n:], axis=1)  
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
        
    @classmethod
    def get_test_data(self, file_name = '../input/PPv4-Predict.csv' ):
        self.data2Predict = pd.read_csv(file_name)
        self.data2Predict.reset_index()
        return self.data2Predict
    

    @classmethod
    def plot_history(self, history):
        loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
        val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
        acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
        val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

        if len(loss_list) == 0:
            print('Loss is missing in history')
            return 

        ## As loss always exists
        epochs = range(1,len(history.history[loss_list[0]]) + 1)

        ## Loss
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        plt.figure(1)
        for l in loss_list:
            plt.plot(epochs, history.history[l], 'b', label='Training loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))
        for l in val_loss_list:
            plt.plot(epochs, history.history[l], 'g', label='Validation loss (' + str(str(format(history.history[l][-1],'.5f'))+')'))

        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        ## Accuracy
        plt.figure(1)
        for l in acc_list:
            plt.plot(epochs, history.history[l], 'r', label='Training accuracy (' + str(format(history.history[l][-1],'.5f'))+')')
        for l in val_acc_list:    
            plt.plot(epochs, history.history[l], 'g', label='Validation accuracy (' + str(format(history.history[l][-1],'.5f'))+')')

        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

    @classmethod
    def save_model(self, model, predict_number):
        # serialize model to JSON
        model_json = model.to_json()
        with open(str(self.algo_number) + '_' + str(predict_number) + "_model.json", "w") as json_file:
            json_file.write(simplejson.dumps(simplejson.loads(model_json), indent=4))

        # serialize weights to HDF5
        model.save_weights(str(self.algo_number) + '_' + str(predict_number) + "_model.h5")
        print("Saved model to disk ", self.algo_number, " Predict #N ", predict_number)

    @classmethod
    def print_result(self, predicted_values ):
        test_df = pd.read_csv('../input/TestResult.csv', sep='\s+', header=None, names=['D','N1','N2','N3','N4','N5','N6','N7'])
        test_df['D'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
        test_df['D'] = pd.to_numeric(test_df['D'])

        cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
        test_df = self.data2Predict.merge(test_df, left_on='T', right_on='D', how='inner')
        test_df = test_df[cols]

        tdfResult = predicted_values.drop(predicted_values.columns[0], axis=1) ;

        
        actual_result = test_df[cols[1:]].values
        predicted_result = np.unique(tdfResult.values)
        predicted_result = [np.unique(a) for a in tdfResult.values]

        matched = self.getIntersection(actual_result, predicted_result)

        c = 0
#         for i in range(len(matched)):
#             print(int(self.data2Predict.loc[i]['T']), '<', len(predicted_result[i]), '> ', actual_result[i], ' **', matched[c], '** ', predicted_result[i])
#             c += 1
#         for i in range(c, len(predicted_result)):
#             print(int(self.data2Predict.loc[i]['T']), ' Predicted: ', predicted_result[i], ' ')

        for i in range(min(len(matched),len(self.data2Predict))):
            print(int(self.data2Predict.loc[i]['T']), '<', len(predicted_result[i]), '> ', actual_result[i], ' **',  actual_result[i], ' ', predicted_result[i], ' ', matched[c])
            c += 1
        for i in range(c, min(len(predicted_result),len(self.data2Predict))):
            print(int(self.data2Predict.loc[i]['T']), ' Predicted: ', predicted_result[i], ' ')

            
    @classmethod
    def load_model(self, predict_number):
        json_file = open(str(self.algo_number) + "_" + str(predict_number)+'_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        # load weights into new model
        loaded_model.load_weights(str(self.algo_number) + "_" + str(predict_number)+"_model.h5")
        print("Loaded model from disk " + str(self.algo_number) + "_" + str(predict_number) + "_model" )
        return loaded_model

    @classmethod
    def getTargets(self):
        return np.array([self.get_result_n(i)['N'] for i in range(1,8)]).T

    @classmethod
    def getTarget(self, N):
        return ([self.get_result_n(i)['N'] for i in range(1,8)])[N-1]


    @staticmethod
    def getIntersection(p1, p2):
        return [reduce(np.intersect1d, (p.astype(int), a.astype(int))) for (p,a) in zip(p1, p2)]

    @classmethod
    def get_test_result(self, result='../input/TestResult.csv'):
        #load the test Toto Results files
        test_df = pd.read_csv(result, sep='\s+', header=None, names=['D','N1','N2','N3','N4','N5','N6','N7'])
        test_df['D'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
        test_df['D'] = pd.to_numeric(test_df['D'])

        #Merge the Planet Position File with the Toto Results file
        test_df = self.data2Predict.merge(test_df, left_on='T', right_on='D', how='inner')

        #Extract only the Results for all dates
        cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
        test_df = test_df[cols]

        actual_result = test_df[cols[1:]].values
        return actual_result
        
    @classmethod
    def get_matched_intersection(self, df_prediction1, df_prediction2):
        intersected_values = (self.getIntersection(np.array(df_prediction1), np.array(df_prediction2)))
        return list(self.getIntersection(self.get_test_result(), intersected_values))
        
    @classmethod
    def getAccuracyCount(self, prediction):
        actual = self.get_test_result()
        matched = []
        if len(prediction) == 0: return 0
        for (p,a) in zip(prediction, actual):
    #        print ( "p: ", p, " a: ", a, (set(p.astype(int)) & set(a)) )
            matched.append((set(p.astype(int)) & set(a)))
        return sum(len(num) > 0 for num in matched)/len(matched)*100.00

    @classmethod
    def print_weighted_numbers(self, dfPredictions):
        matched = []
        for (p,a) in zip(dfPredictions, self.get_test_result()):
    	      matched.append(len(set(p.astype(int)) & set(a)))
        weighted_match = [(1+2*(N/10)) for N in matched]
        return matched, weighted_match

    @staticmethod
    def bins_labels(bins, **kwargs):
        bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
        plt.xlim(bins[0], bins[-1])

    @classmethod
    def plot_matched_counts(self, predicted):
        matched = []
        for (p,a) in zip(predicted, self.get_test_result()):
            matched.append(len(set(p.astype(int)) & set(a)))
        bins = np.arange(8) - 0.5
        plt.hist(matched, bins, rwidth=0.8)
        plt.xticks(range(8))
        plt.xlim([-1, 8])
        plt.show()


    @classmethod
    def print_predictions(self, dfPredictions):
        actual_result = self.get_test_result()
#         #load the test Toto Results files
#         test_df = pd.read_csv(result, sep='\s+', header=None, names=['D','N1','N2','N3','N4','N5','N6','N7'])
#         test_df['D'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
#         test_df['D'] = pd.to_numeric(test_df['D'])

#         #Merge the Planet Position File with the Toto Results file
#         test_df = self.data2Predict.merge(test_df, left_on='T', right_on='D', how='inner')

#         #Extract only the Results for all dates
#         cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
#         test_df = test_df[cols]

        tdfResult = dfPredictions.drop(dfPredictions.columns[0], axis=1) ;

#         actual_result = test_df[cols[1:]].values
        predicted_result = tdfResult.values

        matched = MyTotoResearch.getIntersection(actual_result, predicted_result)

        c = 0
        for i in range(min(len(matched),len(self.data2Predict))):
            print(int(self.data2Predict.loc[i]['T']), ' ', actual_result[i], ' ', predicted_result[i], ' ', matched[c])
            c += 1
        for i in range(c, min(len(matched),len(self.data2Predict))):
            print(int(self.data2Predict.loc[i]['T']), ' Predicted: ', predicted_result[i], ' ')

#        print(dfPredictions)

#         for i in range(len(matched)):
#             print(int(self.data2Predict.loc[i]['T']), ' ', actual_result[i], ' ', predicted_result[i], ' ', matched[c])
#             c += 1
#         for i in range(c, len(predicted_result)):
#             print(int(self.data2Predict.loc[i]['T']), ' Predicted: ', predicted_result[i], ' ')

    @classmethod
    def print_predictionsV2(self, dfPredictions):
        actual_result = self.get_test_result()
#         #load the test Toto Results files
#         test_df = pd.read_csv(result, sep='\s+', header=None, names=['D','N1','N2','N3','N4','N5','N6','N7'])
#         test_df['D'].replace(regex=True,inplace=True,to_replace=r'-',value=r'')
#         test_df['D'] = pd.to_numeric(test_df['D'])

#         #Merge the Planet Position File with the Toto Results file
#         test_df = self.data2Predict.merge(test_df, left_on='T', right_on='D', how='inner')

#         #Extract only the Results for all dates
#         cols = ['D', 'N1','N2','N3','N4','N5','N6','N7']
#         test_df = test_df[cols]

#        tdfResult = dfPredictions.drop(dfPredictions.columns[0], axis=1) ;

#         actual_result = test_df[cols[1:]].values
#        predicted_result = tdfResult.values

        matched = MyTotoResearch.getIntersection(actual_result, dfPredictions)

        c = 0
        for i in range(min(len(matched),len(dfPredictions))):
            print(int(self.data2Predict.loc[i]['T']), ' ', actual_result[i], ' ', dfPredictions[i], ' ', matched[c])
            c += 1
        for i in range(c, min(len(matched),len(dfPredictions))):
            print(int(self.data2Predict.loc[i]['T']), ' Predicted: ', dfPredictions[i], ' ')

#        print(dfPredictions)

            
def getAdjustedDataF(df,f):
    #Use only Planet Positions Testing
    cols = ['L','M','S', 'R','E','A','V' ,'J','U','K']
    X = df[cols]
    deg = f
    
#     X['S_3'] = X['S'] // (deg*3)
#     X['L_3'] = X['L'] // (deg*3)
#     X['M_3'] = X['M'] // (deg*3)
#     X['R_3'] = X['R'] // (deg*3)
#     X['E_3'] = X['E'] // (deg*3)
#     X['A_3'] = X['A'] // (deg*3)
#     X['V_3'] = X['V'] // (deg*3)
#     X['J_3'] = X['J'] // (deg*3)
#     X['U_3'] = X['U'] // (deg*3)


#     X['S_2'] = X['S'] // (deg*2)
#     X['L_2'] = X['L'] // (deg*2)
#     X['M_2'] = X['M'] // (deg*2)
#     X['R_2'] = X['R'] // (deg*2)
#     X['E_2'] = X['E'] // (deg*2)
#     X['A_2'] = X['A'] // (deg*2)
#     X['V_2'] = X['V'] // (deg*2)
#     X['J_2'] = X['J'] // (deg*2)
#     X['U_2'] = X['U'] // (deg*2)

    X['S_1'] = X['S'] // (deg)
    X['L_1'] = X['L'] // (deg)
    X['M_1'] = X['M'] // (deg)
    X['R_1'] = X['R'] // (deg)
    X['E_1'] = X['E'] // (deg)
    X['A_1'] = X['A'] // (deg)
    X['V_1'] = X['V'] // (deg)
    X['J_1'] = X['J'] // (deg)
    X['U_1'] = X['U'] // (deg)
   
    X = X.drop(cols, axis=1)
    return X

print("Done.")
