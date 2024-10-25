
import pandas as pd

def subreddit_dataframe_condense(df):
    """
    Reassigns Pushshift archives to condensed df for annotation, assigns columns for strain, 
    explicit targeting, implicit vulnerability tags
    """
    df = df[[
             'author',
             'created_utc',
             'date',
             'id',
             'num_comments',
             'selftext',
             'subreddit',
             'title',
             ]].copy()

    df.rename(
              columns = {
                         'author': 'p_au',
                         'created_utc': 'p_utc',
                         'date': 'p_date',
                         'id': 'p_id',
                         'num_comments': 'n_cmnt',
                         'selftext': 'text',
                         'subreddit': 'sbrt',
                         'title': 'p_titl',
                         }, inplace = True,
            )

    df = df.assign(
                   asp = ' ',      ### s_1...3 strains
                   asp_rtnl = ' ',
                   dep = ' ',
                   dep_rtnl = ' ',
                   val = ' ',
                   val_rtnl = ' ',
                   prg = ' ',      ### E_1,2 explicit targeting
                   tgd = ' ',
                   age = ' ',      ### I_1...3 implicit vulnerabilities
                   race = ' ',     
                   dbty = ' ',
                   insb = ' ',     ### insubstantial
                   )

    df = df[~df['text'].isin([
                              '[deleted]',
                              '[removed]',
                              ])]

    return df

import pandas as pd

def subreddit_parse(df, col):
    """
    Parses df by subreddit, returns dict 'sub_d' of subreddit-specific df objects.
    """
    uniq_val = df[col].unique()
    sub_d = {}
    for val in uniq_val:
        sub_d[f'd_{val}'] = df[df[col] == val].copy()

    return sub_d
