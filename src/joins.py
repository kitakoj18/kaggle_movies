#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:18:36 2020

@author: kojikitagawa

Join older movie_lens dataset with newer one by movie name -
id in movie_lens is superceded and does not match with other ids

Doing this join to be able to get cast information from newer dataset
"""

import numpy as np
import pandas as pd

import re

def clean_title(title):
    
    # remove year in parenthesis at end of title
    # remove any translations after title name to match with title name in 
    #newer movie_lens dataset
    pattern = ' (\(.*\))? ?(\(\d*?\))'
    title = re.sub(pattern, '', title).strip()
    
    # remove ', The' from the end and add 'The ' to the beginning
    if title[-5:] == ', The':
        title = 'The ' + title[:-5]
        title = title.strip()
        
    return title

def merge_movie_datasets():
    
    df_movies_old = pd.read_csv('../data/movie_lens/movies.csv')
    df_movies_old['title'] = df_movies_old['movie_title'].apply(clean_title)
    
    df_movies = pd.read_csv('../data/movies_metadata.csv')
    df_movies = df_movies[['id', 'title']]
    
    df = df_movies_old.merge(df_movies, left_on='title', right_on='title')
    df.drop_duplicates(subset='title', inplace=True)

    return df    
    