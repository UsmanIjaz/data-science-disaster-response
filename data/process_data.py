import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from langdetect import detect


def load_data(messages_filepath, categories_filepath):
    """
    Load the csv files and combine them into a data frame
    --
    Inputs:
        messages_filepath: csv file contains messages
        categories_filepath: csv file contains categories
    Outputs:
        df: the combined dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories, how='inner',on=['id'])
    
    return df

def lang_detect(txt):
    """
    Detect the language of a given text using langdetect library
    --
    Inputs:
        txt: message string
    Outputs:
        string: language name string or None if language not detected
    """
    try:
        return detect(txt)
    except:
        return None

def clean_data(df):
    """
    Clean the dataframe. It includes removing duplicates and non-English messages 
    --
    Inputs:
        df: messages and categories combined dataframe dataframe
    Outputs:
        df: cleaned dataframe
    """    
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x:x.split("-")[1])
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    categories.related[categories.related==2] = 0
    df.drop(['categories'], axis=1, inplace=True)
    
    df['message_lang'] = df.message.apply(lambda x: lang_detect(x))
    df['original_lang'] = df.original.apply(lambda x: lang_detect(x))
    df.message[df.original_lang.isin(['en']) & ~df.message_lang.isin(['en'])] = df.original
    df.message_lang[df.original_lang.isin(['en']) & ~df.message_lang.isin(['en'])] = "en"
    df_eng = df[df.message_lang=="en"]
    df_eng = df[df.message_lang.notnull()]
    df_eng = df_eng.drop(['message_lang','original_lang'], axis=1)
    
    df_eng = pd.concat([df_eng, categories], axis=1)
    df_eng = df_eng[df_eng.id.duplicated()==False]
    
    return df_eng


def save_data(df, database_filename):
    """
    Saves the dataframe into a sqlite database 
    --
    Inputs:
        df: messages and categories combined dataframe dataframe
        database_filename: Name of the database file
    """    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, if_exists='replace',index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()