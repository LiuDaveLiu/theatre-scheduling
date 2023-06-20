import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

class RoomClassifier:
    def __init__(self, data_file_path):
        """
        Read case and session data into Pandas DataFrames
        Args:
            case_file_path (str): path to case data in CSV format
        """
        try:
            self.df = pd.read_csv(data_file_path)
        except FileNotFoundError:
            print("Case data not found.")
        self.df = self.df[['Scheduled Setup Start', 'Scheduled Cleanup Complete', 'Room', 'Location', 'Primary Surgeon Name', 'Primary Service','Primary Procedure CPT Code']]
        self.df['Scheduled Setup Start']=pd.to_datetime(self.df['Scheduled Setup Start'])
        self.df['Scheduled Cleanup Complete']=pd.to_datetime(self.df['Scheduled Cleanup Complete'])
        self.df['Scheduled Room Duration']=(self.df['Scheduled Cleanup Complete']-self.df['Scheduled Setup Start']).astype('timedelta64[m]')
        self.df['test']=1
        self.df = self.df.dropna()
        self.df['Primary Procedure CPT Code']=self.df['Primary Procedure CPT Code'].str.replace('\D+','').astype(int)
        
        self.df = self.df[self.df['Room'].str.contains(r'\d', regex=True)]

        self.df = self.df[self.df['Location']=='MAYS OR']
        self.df['Room']=self.df.apply(lambda row: int(re.search(r'\d+', row['Room']).group()), axis = 1)
        self.df = self.df.loc[self.df['Room']<41]
        
        self.df=self.df.assign(CaseID=range(len(self.df)))

        self.model = self.create_model()
    
    def create_model(self):
        X = self.df[['Primary Surgeon Name', 'Primary Service', 'Scheduled Room Duration', 'Primary Procedure CPT Code']]

        y = self.df['Room']

        features_to_encode = ['Primary Surgeon Name', 'Primary Service']
        
        col_trans = make_column_transformer((OneHotEncoder(), features_to_encode), remainder = "passthrough")

        seed=1
        rf_classifier = RandomForestClassifier(min_samples_leaf=50, n_estimators=150, bootstrap=True, oob_score=True, n_jobs=-1, random_state=seed, max_features='auto')
        
        pipe = make_pipeline(col_trans, rf_classifier)
        pipe.fit(X, y)
        return pipe
    
    def solve(self, test_path):

        df = pd.read_csv(test_path)

        df = df[['Scheduled Setup Start', 'Scheduled Cleanup Complete', 'Room', 'Primary Surgeon Name', 'Primary Service','Primary Procedure CPT Code']]
        df['Scheduled Setup Start']=pd.to_datetime(df['Scheduled Setup Start'])
        df['Scheduled Cleanup Complete']=pd.to_datetime(df['Scheduled Cleanup Complete'])
        df['Scheduled Room Duration']=(df['Scheduled Cleanup Complete']-df['Scheduled Setup Start']).astype('timedelta64[m]')
        df['test']=1
        df = df.dropna()
        
        df = df[df['Room'].str.contains(r'\d', regex=True)]
        df['Room']=df.apply(lambda row: int(re.search(r'\d+', row['Room']).group()), axis = 1)
        df = df.loc[df['Room']<41]
        
        df=df.assign(CaseID=range(len(df)))
        
        X_test = df[['Primary Surgeon Name', 'Primary Service', 'Scheduled Room Duration', 'Primary Procedure CPT Code']]

        return self.model.predict_proba(X_test)  