#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:36:18 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

import ast

# script to extract and format needed data from various datasets
df_creds = pd.read_csv('../data/subsets/credits.csv')
df_movies = pd.read_csv('../data/subsets/movies_metadata.csv')

def convert_to_list(str_list):
    '''
    Convert strings into array literals
    '''
    
    converted_list = ast.literal_eval(str_list)
    return converted_list

# get cast of movie
def get_cast(casts):
    
    casts = convert_to_list(casts)
    cast_list = []
    
    for cast in casts:
        cast_list.append(cast['name'])
        
    return cast_list

# get director and producer of movie
def get_crew(crews):
    
    crews = convert_to_list(crews)
    crew_list = []
    
    for crew in crews:
        crew_job = crew['job']
        if crew_job == 'Director' or crew_job == 'Executive Producer':
            crew_list.append(crew['name'])
            
    return crew_list

def credit_vectorize(creds):
    
    count_vec = CountVectorizer(analyzer=lambda x:x)
    X = count_vec.fit_transform(creds)
    
    credit_counts = [vec for vec in X]
    
    return credit_counts


def get_movie_creds():
    
    df_creds = pd.read_csv('../data/subsets/credits.csv')
    
    df_creds['cast'] = df_creds['cast'].apply(get_cast)
    df_creds['crew'] = df_creds['crew'].apply(get_crew)
    
    cast_counts = credit_vectorize(df_creds['cast'])
    crew_counts = credit_vectorize(df_creds['crew'])
    
    df_creds['cast_count'] = cast_counts
    df_creds['crew_count'] = crew_counts
    
    df_creds = df_creds[['id', 'cast_count', 'crew_count']]
    
    return df_creds