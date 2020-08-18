import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

def clean_for_nmf(df,*, max_nas = 50):
    '''Clean data frame for NMF.  
    (i) Removes non-numeric and negative features
    (ii) deletes features with more thatn max_nas nan/inf values 
    (iii) replaces missing values with feature median
    (iv) normalizes columns
    (v) sorts by FIPS
    '''
    
    clean_df = df.select_dtypes([np.number])
    clean_df = clean_df.replace(np.inf, np.nan)
    clean_df = clean_df.loc[:, (df.isnull().sum(axis=0) <= max_nas)]
    clean_df = clean_df.fillna(clean_df.median(axis=0))
    clean_df = clean_df.loc[:, (clean_df > 0).all('rows')]
    clean_df = pd.DataFrame(normalize(clean_df, axis=0), 
                  columns=clean_df.columns, index=clean_df.index)
    clean_df = clean_df.sort_index()
    return clean_df

def read_yu_data():
    '''Read county demographic data into dataframe with contyFIPS as index'''
    
    data = pd.read_csv('yu_data_NO_AHRF.csv')
    data.set_index('countyFIPS', inplace=True)
    return data

def get_covid_column(feature):
    '''Returns normalized column from jh_covid.csv indexed by FIPS code'''
    covid_data = pd.read_csv('jh_covid.csv')
    covid_data.set_index('FIPS', inplace=True)
    covid_col = covid_data[[feature]]
    covid_col = covid_col.reset_index().dropna().set_index('FIPS')
    covid_col = pd.DataFrame(normalize(covid_col, axis=0), columns=covid_col.columns, index=covid_col.index)
    covid_col = covid_col.sort_index()
    return covid_col

def normalize_columns(df):
    return pd.DataFrame(normalize(df, axis=0), columns=df.columns, index=df.index)

def training_test_data(demo_data, covid_col, test_frac=0.2):
    
    demo_clean = clean_for_nmf(demo_data, max_nas=100)
    
    covid_merged = pd.concat([covid_col, demo_clean], axis=1)
        
    #ignore counties that are left out of either data set
    covid_merged = covid_merged.loc[covid_merged.notnull().all(axis=1), :] 
        
    merged_train, merged_test = train_test_split(covid_merged, test_size=test_frac)    
    
    demo_train = pd.DataFrame(merged_train.iloc[:, 1:])    
    covid_train = pd.DataFrame(merged_train.iloc[:,0])
    
    demo_test = pd.DataFrame(merged_test.iloc[:, 1:])
    covid_test = pd.DataFrame(merged_test.iloc[:, 0])
        
    #ensure that test are training data are normalized separately
    clean_train = normalize_columns(demo_train)
    clean_test = normalize_columns(demo_test)
    

    return (demo_train,covid_train), (demo_test, covid_test)