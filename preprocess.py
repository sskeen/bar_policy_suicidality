
import pandas as pd

def augment_training_data_with_rationales(df):
    """
    Identifies all pos_1 strn, duplicates as new row below, replaces new row 'text' with appended concatenated asp-dep-val 'rtnl'.
    """
    augmented_rows = []

    for index, row in df.iterrows():
        augmented_rows.append(row)

        if row['strn'] == 1:
            duplicate_row = row.copy()
            duplicate_row['text'] = duplicate_row['rtnl']
            augmented_rows.append(duplicate_row)

    df_augmented = pd.DataFrame(augmented_rows)

    return df_augmented

import pandas as pd

def dummy_code_augmented_rows(df):
    """
    Identifies all rationale-augmented rows in df, dummy codes for deletion prior to evaluation.
    """
    df = df.reset_index(drop = True)

    df['aug'] = 0

    for i in range(1, len(df)):
        if df.at[i, 'rtnl'] != '.' and df.at[i, 'rtnl'] == df.at[i-1, 'rtnl']:
            df.at[i, 'aug'] = 1

    return df

import os
import pandas as pd

def read_and_append_jsonl_posts(directory, chunk_size = 10000):
    """
    Reads and appends JSONL posts archives from Arctic Shift archives dir.
    """
    d_posts = pd.DataFrame()
    
    for filename in os.listdir(directory):
        if filename.endswith("_posts.jsonl"):
            filepath = os.path.join(
                                    directory, 
                                    filename,
                                    )
            
            for chunk in pd.read_json(
                                      filepath, 
                                      lines = True, 
                                      chunksize = chunk_size,
                                      ):
              
                d_posts = pd.concat([d_posts, chunk], ignore_index = True)

    return d_posts

import os
import pandas as pd

def read_and_append_jsonl_comments(directory, chunk_size = 10000):
    """
    Reads and appends JSONL comments archives from Arctic Shift archives dir.
    """
    d_comments = pd.DataFrame()
    
    for filename in os.listdir(directory):
        if filename.endswith("_comments.jsonl"):
            filepath = os.path.join(
                                    directory, 
                                    filename,
                                    )
            
            for chunk in pd.read_json(
                                      filepath, 
                                      lines = True, 
                                      chunksize = chunk_size,
                                      ):
              
                d_comments = pd.concat([d_comments, chunk], ignore_index = True)

    return d_comments
