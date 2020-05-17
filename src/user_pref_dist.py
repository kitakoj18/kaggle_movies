#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:02:18 2020

@author: kojikitagawa
"""

import numpy as np
import pandas as pd

from scipy.stats import entropy

from movie_sims import make_sql_table

genre_cols = ['unknown', 'action', 'adventure', 'animation', 'childrens',
                  'comedy', 'crime', 'documentary', 'drama', 'fantasy', 'film-noir', 
                  'horror', 'musical', 'mystery', 'romance', 'sci-fi', 'thriller',
                  'war', 'western']

cast_col = 'cast_count'
crew_col = 'crew_count'


def create_user_entropy(df_user_pref, table_name):
    '''
    Calculates the entropy for each user and their preference vector (genre, cast, or crew)
    to get a value for how evenly or unevenly a user's preference distribution is
    and creates new SQL table with user_id and the entropy value
    '''
    
    df_user_pref = df_user_pref.copy()
    
    if table_name == 'user_genre_ent':
        cols = genre_cols
    elif table_name == 'user_cast_ent':
        cols = cast_col
    else:
        cols = crew_col
    
    entropies = []
    for idx, row in df_user_pref.iterrows():

        if table_name == 'user_genre_ent':
            user_vec = row[cols].to_numpy()
        else:
            user_vec = row[cols].toarray().flatten()

        ent = entropy(user_vec)
        entropies.append(ent)
        
    df_user_pref['entropy'] = entropies
    
    # for cast table, make sure crew_count col is dropped too and vice versa
    if table_name == 'user_cast_ent':
        cols = [cols, 'crew_count']
    if table_name == 'user_crew_ent':
        cols = [cols, 'cast_count']
        
    df_user_pref.drop(columns=cols, inplace=True)
    
    make_sql_table(df_user_pref, table_name) 
    
def get_entropy(user_id, table_name, conn):
    '''
    Get user's entropy for their genre, cast, or crew preference vector
    '''
    
    query = '''
            SELECT * 
            FROM {}
            WHERE user_id = {}
            '''.format(table_name, user_id)
    
    result = conn.execute(query)
    entropy = result.fetchone()['entropy']
    
    return entropy
    
    
    
    