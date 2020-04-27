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


def generate_unwatched_movies(user_id, user_movie_set, movie_set, num_unwatched):
    '''
    Helper function for user_watched_movies to randomly select num_unwatched number of movies 
    that the user has not watched based off of their set of movies they have watched
    '''
    unwatched_movie_ids = []
    num_movies = len(movie_set)
    
    while num_unwatched > 0: 
        
        # randomly select a movie id from movie_set
        rand_movie_idx = random.randint(0, num_movies-1)
        # if selected movie id is in user_movie_set, user watched movie so continue
        if movie_set[rand_movie_idx] in user_movie_set:
            continue
        
        unwatched_movie_ids.append(movie_set[rand_movie_idx])
        num_unwatched -= 1

    df_unwatched = pd.DataFrame({'movie_id': unwatched_movie_ids})
    df_unwatched['user_id'] = user_id
    df_unwatched['watched'] = 0
    #reorder columns to match column order of dataframe it's being appended to
    df_unwatched = df_unwatched[['user_id', 'movie_id', 'watched']]
    
    return df_unwatched

def users_watched_movies(num_unwatched=5):
    '''
    Function that generates table with all users and movies they did watch
    and predetermined number of movies they did not watch to use as a base table to create final dataset
    '''
    df_ratings = pd.read_csv('../data/movie_lens/ratings_subset.csv')
    # only need these two columns for now - will replace rating column later with predicted rating
    df_movies_watched = df_ratings[['user_id', 'movie_id']]
    # add column to indicate movie was watched by user
    df_movies_watched['watched'] = 1
    
    # get set of movies watched for each user
    df_user_movie_set = df_movies_watched.groupby('user_id', sort=False) \
                                    .agg({'movie_id': lambda x: set(x)}) \
                                    .reset_index()
               
    # get set of unique movie_ids from movies we have information on                     
    df_movies_info = join_credits()
    movie_set = df_movies_info['movie_id'].unique()
    
    # for each user, select num_watched number of movies NOT watched by user
    for idx, row in df_user_movie_set.iterrows():
        
        user_id = row['user_id']
        user_movie_set = row['movie_id']
        
        # get movies not watched by user and append to df_watched
        df_unwatched = generate_unwatched_movies(user_id, user_movie_set, movie_set, num_unwatched)
        df_movies_watched = df_movies_watched.append(df_unwatched, ignore_index=True)
        
    return df_movies_watched


def get_base_tables():
    '''
    Generate and return base tables to generate final dataset
    '''
    
    df_movie_info = join_credits()
    df_movies_watched = users_watched_movies
    
    return df_movie_info, df_movies_watched
    