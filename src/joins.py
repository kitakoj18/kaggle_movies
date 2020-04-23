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
    with newer movie ids
    '''
    
    df_movies_old = pd.read_csv('../data/movie_lens/movies.csv')
    df_movies_old['title'] = df_movies_old['movie_title'].apply(clean_title)
    
    df_movies_new = pd.read_csv('../data/movies_metadata.csv')
    df_movies_new = df_movies_new[['id', 'title']]
    
    df_movie_info = df_movies_old.merge(df_movies_new, left_on='title', right_on='title')
    df_movie_info.drop_duplicates(subset='title', inplace=True)

    return df_movie_info
    
def join_genres():
    '''
    Function to join movie genres from movies dataset with ratings dataset
    df_movies_watched will be a base table used to create final dataset
    '''
    
    df_ratings = pd.read_csv('../data/movie_lens/ratings.csv')
    df_movie_info = join_movie_datasets()
    
    genre_cols = ['movie_id', 'id', 'unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']
    df_genres = df_movie_info[genre_cols]
    
    # inner join ratings with genres - only want ratings where we have updated id information
    df_movies_watched = df_ratings.merge(df_genres, left_on='movie_id', right_on='movie_id')
    # print(df_movies_rated.info())
    
    return df_movies_watched


def join_credits():
    '''
    Function that joins df_movies_watched to df_creds from cast_crew.py, which
    contains cast and crew count
    '''
    df_credits = get_movie_creds()
    df_movies_watched = join_genres()
    
    df_movies_watched = df_movies_watched.merge(df_credits, left_on='id', right_on='id')
    
    return df_movies_watched

def get_movies_watched():
    
    
    