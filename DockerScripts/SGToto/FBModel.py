import pandas as pd
import numpy as np
import pickle
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder, StandardScaler


class FBModel:
    def __init__(self):
        np.random.seed(42)
        pass
    
    def get_df(self):
        return self.df
    
    def preprocess (self, df):
        self.df = df 
        #nans = lambda df: df[df.isnull().any(axis=1)]
        #show rows which has nan in any columns
        #nans(df)
        
        #delete missing messages
        delete_rows = df[df['message'].isna()].index
        df.drop(index=delete_rows, inplace=True)
        # df['click_rate'] = df['clicks_unique'] / (df['impressions_organic_unique'] + df['impressions_paid_unique'])
        # df['reaction_rate'] = (df['reactions_haha'] + df['reactions_like'] + df['reactions_love'] + df['reactions_sorry']) / (df['impressions_organic_unique'] + df['impressions_paid_unique'])
        # df['engagement_rate'] = df['engagements'] / (df['impressions_organic_unique'] + df['impressions_paid_unique'])
        # df['comment_rate'] = df['comments'] / (df['impressions_organic_unique'] + df['impressions_paid_unique'])
        df['shares'] = df['comments'] / (df['impressions_organic_unique'] + df['impressions_paid_unique'])

        self.encode_features(df)
        
    def encode_features(self, df):
        cols_to_encode = ['amplification', 'intent', 'type']
        self.encoders = {}
        for i in range(0, len(cols_to_encode)):
            col = cols_to_encode[i]
            self.encoders[col] = LabelEncoder()
            df[col + '_encoded'] = self.encoders[col].fit_transform(df[col])
        
    def scale_features(self, X):
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        return self.scaler.transform(X)
    
    def get_features_to_use(self, df):
        features = [
            'amplification_encoded', 
#    'intent_encoded', 
            'type_encoded', 
#     'last_post_time_diff',
#    'time', 
            'weekday',
#    'num_creatives',
            'message_length',
        ]
        return df[features]
    
    
    def save_model(self, model, file_name_without_extension):
        model_json = model.to_json()
        with open(file_name_without_extension + ".json", "w") as json_file:
            json_file.write(model_json)
            # serialize weights to HDF5
        model.save_weights(file_name_without_extension + ".h5")
        print("Saved model to disk")
        pickle.dump(self.scaler, open(file_name_without_extension + '-scaler.pkl', 'wb'))
        pickle.dump(self.encoders, open(file_name_without_extension + '-encoders.pkl', 'wb'))


    def load_model(self, file_name_without_extension):
        # fix random seed for reproducibility
        # load json and create model
        json_file = open(file_name_without_extension + '.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(file_name_without_extension + ".h5")
        self.scaler_loaded = pickle.load(open(file_name_without_extension + '-scaler.pkl', 'rb'))
        self.encoders_loaded = pickle.load(open(file_name_without_extension + '-encoders.pkl', 'rb'))
        return model
    
    def get_message_doc(self, messages):
        # integer encode the messages
        vocab_size = 50
        encoded_docs = [one_hot(d, vocab_size) for d in messages]

        # pad messages to a max length of 4 words
        max_length = 4
        return  pad_sequences(encoded_docs, maxlen=max_length, padding='post')


    def predict ( self, df, file_name_without_extension ):
        model = self.load_model(file_name_without_extension)
        cols_to_encode = ['amplification', 'intent', 'type']        
        for col in cols_to_encode:
            df[col + '_encoded'] = self.encoders_loaded[col].transform(df[col])
        X = self.get_features_to_use(df)
        X_scaled = self.scaler_loaded.transform(X)
        padded_docs = self.get_message_doc(df['message'].values)
        return model.predict([padded_docs,X_scaled])
        