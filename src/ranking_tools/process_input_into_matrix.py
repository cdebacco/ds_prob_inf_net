'''
Process input csv into a tensor to be input to any SR model variant
**Copyright**: Caterina De Bacco, 2025
'''

import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import calendar

def process_games(
        df: pd.DataFrame,
        ht: str = 'home_team',
        at: str = 'away_team'
    )-> pd.DataFrame:
    '''
    Process games
    '''
    df.loc[:, 'score_diff'] = df['home_score'] - df['away_score']
    df.loc[:,'home_points'] = 1
    df.loc[:, 'away_points'] = 1
    mask = df['score_diff'] > 0
    df.loc[mask, 'home_points'] = 3
    df.loc[mask, 'away_points'] = 0
    mask = df['score_diff'] < 0
    df.loc[mask, 'home_points'] = 0
    df.loc[mask, 'away_points'] = 3

    df = keep_connected_component(df,ht=ht,at=at)

    return df


def keep_connected_component(df: pd.DataFrame,
                             ht: str = 'home_team',
                             at: str = 'away_team')-> pd.DataFrame:
    '''
    Keep only largest connected component
    '''
    g = nx.from_pandas_edgelist(df, ht,at, create_using=nx.Graph)
    nodes_lcc = list(nx.connected_components(g))[0]

    if nx.number_connected_components(g) > 1:
        print(list(nx.connected_components(g))[1:])

    cond1 = df[ht].isin(nodes_lcc)
    cond2 = df[at].isin(nodes_lcc)
    mask = cond1 | cond2
    return df[mask]

def df2matrix(
        df: pd.DataFrame = None,
        path_encoder: str = None,
        method: str ='points',
        indir: str = "../../../data/output/matches/scores/",
        filename: str = 'tot_score.csv',
        score_label: str = 'points',
        ht: str = 'home_team',
        at: str ='away_team',
        verbose: bool = False
    ):
    '''
    Gets a pd.DataFrame and transforms into tensor and encoder for teams
    The DataFrame should be already preprocessed into integer time intervals
    '''
    if df is None:
        df = pd.read_csv(f"{indir}{filename}")
    assert ht in df, f"column {ht} not in df!\nAvailable = {df.columns}"
    assert at in df, f"column {at} not in df!\nAvailable = {df.columns}"

    available_methods = ['diff','score','points']
    assert method in available_methods
    if path_encoder is None:
        teams = list(set(df[ht]).union(set(df[at])))
        encoder_teams = LabelEncoder()
        encoder_teams.fit(teams)
    else:
        encoder_teams = LabelEncoder()
        encoder_teams.classes_ = np.load(path_encoder)

    N = len(set(df[ht])|set(df[at]))
    A = np.zeros((N,N))
    for idx, row in df.iterrows():
        if method == 'diff':
            if row[score_label] > 0:
                A[int(encoder_teams.transform([row[ht]])),int(encoder_teams.transform([row[at]]))] += abs(row[score_label])
            elif row[score_label] < 0:
                A[int(encoder_teams.transform([row[at]])),int(encoder_teams.transform([row[ht]]))] += abs(row[score_label])
            else:
                if verbose == True: print(f'tie between {row[ht]} vs {row[at]} = {row[score_label]:.2f}')
        elif method in ['score','points']:
            A[int(encoder_teams.transform([row[ht]])),int(encoder_teams.transform([row[at]]))] += row[f"home_{score_label}"]
            A[int(encoder_teams.transform([row[at]])),int(encoder_teams.transform([row[ht]]))] += row[f"away_{score_label}"]

    return A,encoder_teams


