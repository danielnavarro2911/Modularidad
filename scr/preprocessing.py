import pandas as pd

def extract_features(df):
    df = df.dropna()

    df.loc[:,'nameDestType'] = df['nameDest'].map(lambda x: 1 if x[0] == 'C' else 0)
    df.loc[:,'nameDestCount'] = df['nameDest'].map(df['nameDest'].value_counts())
    df.loc[:,'nameOrigCount'] = df['nameOrig'].map(df['nameOrig'].value_counts())

    df = df.merge(df[['nameDest', 'amount']].groupby('nameDest').sum().rename(columns={'amount': 'totalReceived'}),
                  how='left', on='nameDest')
    df = df.merge(df[['nameOrig', 'amount']].groupby('nameOrig').sum().rename(columns={'amount': 'totalSent'}),
                  how='left', on='nameOrig')

    df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
    return df

def load_and_process_data(path):
    df = pd.read_csv(path)
    df = extract_features(df)
    X = df.drop('isFraud', axis=1)
    X = pd.get_dummies(X)
    y = df['isFraud']
    return X, y
