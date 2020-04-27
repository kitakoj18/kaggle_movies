#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:18:36 2020

@author: kojikitagawa

"""

import numpy as np
import pandas as pd

import re

import random

from cast_crew import *

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

def join_movie_datasets():
    '''
    Join old MovieLens dataset with newer one on movie titles to match older dataset
    with newer movie ids to later be able to join cast and crew data
    '''
    
    df_movies_old = pd.read_csv('../data/movie_lens/movies_subset.csv')
    df_movies_old['title'] = df_movies_old['movie_title'].apply(clean_title)
    
    df_movies_new = pd.read_csv('../data/subsets/movies_metadata.csv')
    df_movies_new = df_movies_new[['id', 'title']]
    
    df_movie_info = df_movies_old.merge(df_movies_new, left_on='title', right_on='title')
    df_movie_info.drop_duplicates(subset='title', inplace=True)

    return df_movie_info


def join_credits():
    '''
    Function that pulls cast and crew movie data from cast_crew.py
    to include in df_movie_info 
    '''
    
    df_movie_info = join_movie_datasets()
    df_movie_creds = get_movie_creds()
    
    df_movie_info = df_movie_info.merge(df_movie_creds, left_on='id', right_on='id')
    
    return df_movie_info


def get_movie_info():
    
    df_movie_info = join_credits()
    df_movie_info.drop(columns=['release_dt', 'vid_release_dt', 'IMDb_url'], inplace=True)
    return df_movie_info
    