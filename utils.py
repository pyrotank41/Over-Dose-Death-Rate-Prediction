# -*- coding: utf-8 -*-
"""milestones 1, 2, &3.ipynb

Original file is located at
    https://colab.research.google.com/drive/1otAPw-SGgQ568io10qxsWIEq0nSjdXGd

"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

def get_data(location_pair): 
    # Tries to get local version and then defaults to google drive version
    (loc, gdrive)=location_pair
    try:
        out=pd.read_csv(loc)

    except FileNotFoundError:
        print(f"Local file named {loc} not found, downloading from Google Drive...")
        out = download_remote_data(loc, gdrive)
    
    return out

def download_remote_data(file_name, gdrive_file_id):
    "downloads a file from google drive and saves it to local storage as file_name"
    remote_loc = 'https://drive.google.com/uc?export=download&id=' + gdrive_file_id.split('/')[-2]
    out=pd.read_csv(remote_loc)
    print(f"Saving the downloaded file to local storage as {file_name}...")
    out.to_csv(file_name)
    print("... saved.")
    return out 

def drop_colum_with_only_one_unique_value(data):

    # Dropping the columns that have only 1 unique value (no added information from these columns):
    for col in data.columns:
        if len(data[col].unique())==1:
            data.drop(col,axis=1,inplace=True)

    return data

def drug_overdose_data_preprocess(data):
    # Drop unnecessary columns and drop columns with only one unique value
    data.drop(['Footnote',
                'Footnote Symbol', 
                'Percent Pending Investigation',
                'Predicted Value'],
                axis=1,inplace=True)

    data = drop_colum_with_only_one_unique_value(data)
    data.dropna(inplace=True)

    # Drop rows with 'YC' in the 'State' column:
    data = data[data.State != 'YC']

    # we are interested in the data for 2021 only
    data = data[data.Year == 2021]

    # In the 'Indicator' column, we are interested in the rows that have 'Number of Drug Overdose Deaths' in them
    # and we are not interested in the rows that have 'Percent with drugs specified' in them
    data = data[data.Indicator != 'Percent with drugs specified']
    data = data[data.Indicator == 'Number of Drug Overdose Deaths']

    # Drop the 'Indicator' column since we are not interested in it anymore
    data.drop(['Indicator', "Month", "Year", "State"], axis=1, inplace=True)

    # finally get the sum of the drug overdose deaths for each state
    data = data.replace(',','', regex=True)
    data['Data Value'] = data['Data Value'].astype(int)
    data = data.groupby('State Name', as_index=False)['Data Value'].sum()
    data.rename(columns={'Data Value': 'Overdose Deaths'}, inplace=True)

    return data

def final_preprocess(predictors, drug_overdose_data):

    """
    We added more data and summarized the drug overdose deaths. We decided to use data from 2018 since that was easieast and most comprehensive year to obtain data from various data sources.
    Our data now consists of:
    
    *   Tax Rate
    *   List item
    *   Unemployment Rate
    *   Cost of Living Compared to the Average
    *   High School Graduation Rate
    *   GDP Per Capita
    *   Life Expectancy
    *   Average Age
    *   Poverty Rate
    *   Violent Crime Rate
    *   Population
    *   Drug Overdose Deaths

    for all 50 states.
    """


    final_data = predictors.merge(drug_overdose_data).rename(columns={'Data Value': 'Overdose Deaths'})

    # Convert 'GDP Per Capita' and 'Population' from 'Object' type to 'int':
    final_data = final_data.replace(',','', regex=True)
    final_data[['GDP Per Capita', 'Population']] = final_data[['GDP Per Capita', 'Population']].apply(pd.to_numeric)

    # Instead of dropping State Name, convert it to numerical:
    # final_data = final_data.drop(axis=1, labels=['State Name'])

    final_data['State Name'] = pd.Categorical(final_data['State Name'])
    # print(final_data['State Name'].dtypes)
    cat_columns = final_data.select_dtypes(['category']).columns
    final_data[cat_columns] = final_data[cat_columns].apply(lambda x: x.cat.codes)

    # scale the label column "Overdose Deaths" by dividing by 10000:
    final_data["Overdose Deaths"] = final_data["Overdose Deaths"]/10000
    final_data = final_data.rename({'Overdose Deaths': 'Overdose Deaths e10-4'}, axis=1)
    return final_data

def visualize_correlation(data):
    """
    Scatterplot Visualization
    Here we visualize the correlation in a scatterplot to better see the data.
    """
    sns.set_palette('colorblind')
    sns.pairplot(data=data, height=3)
    plt.show()

def get_dataframe(data_pickle_file_name):
    
    try:
        data = pd.read_pickle(data_pickle_file_name)
        print("Dataframe loaded from pickle file.")

    except FileNotFoundError:
        print("Dataframe pickle file not found, creating dataframe and pickleing the file for future use...")

        local_file_name = 'VSRR_Provisional_Drug_Overdose_Death_Counts.csv'
        remote_file_name = 'https://drive.google.com/file/d/1ah5KesrTEmAKraaZM4HSPRhWMihc36gS/view?usp=sharing'

        drug_overdose_data = drug_overdose_data_preprocess(get_data((local_file_name, remote_file_name)))
        
        local_file_name = '2018predictors.csv'
        remote_file_name = 'https://drive.google.com/file/d/1WEDm7_xasrjbV6WsWw773Q1eWa5qaWMU/view?usp=sharing'

        data = final_preprocess(get_data((local_file_name, remote_file_name)), drug_overdose_data)
        data.to_pickle(data_pickle_file_name)

    return data

def main():

    data = get_dataframe('data.pickle')



if __name__ == "__main__":
    main()