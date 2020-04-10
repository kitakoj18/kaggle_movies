#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:36:18 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

import ast

# script to extract and format needed data from various datasets
df_creds = pd.read_csv('../data/subsets/credits.csv')
df_movies = pd.read_csv('../data/subsets/movies_metadata.csv')

def convert_to_list(str_list):
    
    converted_list = ast.literal_eval(str_list)
    return converted_list

# get cast of movie
def get_cast(creds):
    
    creds = convert_to_list(creds)
    cast_list = []
    
    for cred in creds:
        cast_list.append(cred['name'])
        
    return cast_list

# get director and producer of movie
def get_crew(creds):
    
    creds = convert_to_list(creds)
    crew_list = []
    
    for cred in creds:
        crew_job = cred['job']
        if crew_job == 'Director' or crew_job == 'Executive Producer':
            crew_list.append(cred['name'])
            
    return crew_list

# get genre(s) of movie
def get_genres(genres):
    
    genres = convert_to_list(genres)
    genre_list = []
    
    for genre in genres:
        genre_list.append(genre['name'])
        
    return genre_list