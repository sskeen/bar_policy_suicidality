
import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calculate_kappa_by_cycle(cycle_num):
    """
    Calculate Cohen's Kappa and encode disagreements between independent annotators across multiple cycles.

    Parameters:
    -----------
    cycle_num : int
        Annotation cycle number, used to load the corresponding Excel files (e.g., cycle 0, cycle 1).

    Returns:
    --------
    d : pd.DataFrame
        Processed df after merging, includes encoded disagreements in *_dis columns.

    kappa_results : dict
        A dictionary containing the Cohen's Kappa scores for each indepednently co-annotated target.
    """
    # read independently annotated files

    d_sd = pd.read_excel(f'd_cycle{cycle_num}_sd.xlsx', index_col = [0])
    d_sd.columns = [f'{col}_sd' for col in d_sd.columns]

    d_ss = pd.read_excel(f'd_cycle_{cycle_num}_ss.xlsx', index_col = [0])
    d_ss.columns = [f'{col}_ss' for col in d_ss.columns]

    # merge

    d = pd.merge(
                 d_sd,
                 d_ss,
                 left_index = True,
                 right_index = True,
                 )

    # housekeeping

    targets = [
               'asp_sd', 'asp_ss',
               'dep_sd', 'dep_ss',
               'val_sd', 'val_ss',
               'prg_sd', 'prg_ss',
               'tgd_sd', 'tgd_ss',
               'age_sd', 'age_ss',
               'race_sd', 'race_ss',
               'dbty_sd', 'dbty_ss',
               'insb_sd', 'insb_ss',
              ]

    texts = [
             'text_sd', 'text_ss',
             'asp_rtnl_sd', 'asp_rtnl_ss',
             'dep_rtnl_sd', 'dep_rtnl_ss',
             'val_rtnl_sd', 'val_rtnl_ss',
             ]

    d[targets] = d[targets].apply(
                                  pd.to_numeric,
                                  errors = 'coerce',
                                  )
    d[targets] = d[targets].fillna(0)
    d[texts] = d[texts].replace(' ', '.')

    d = d[[
           'p_id_sd', 'p_id_ss', ### sense-check for bad merge
           'text_sd',
           'asp_sd', 'asp_ss',
           'asp_rtnl_sd', 'asp_rtnl_ss',
           'dep_sd', 'dep_ss',
           'dep_rtnl_sd', 'dep_rtnl_ss',
           'val_sd', 'val_ss',
           'val_rtnl_sd', 'val_rtnl_ss',
           'prg_sd', 'prg_ss',
           'tgd_sd', 'tgd_ss',
           'age_sd', 'age_ss',
           'race_sd', 'race_ss',
           'dbty_sd', 'dbty_ss',
           'insb_sd', 'insb_ss',
           ]].copy()

    d.rename(
             columns = {
                        'text_sd': 'text',
                        }, inplace = True,
            )

    # kappa Fx

    def calculate_kappa(d, col_sd, col_ss):
        return cohen_kappa_score(d[col_sd], d[col_ss])

    col_pairs = [
                 ('asp_sd', 'asp_ss'),
                 ('dep_sd', 'dep_ss'),
                 ('val_sd', 'val_ss'),
                 #('prg_sd', 'prg_ss'),
                 #('tgd_sd', 'tgd_ss'),
                 #('age_sd', 'age_ss'),
                 #('race_sd', 'race_ss'),
                 #('dbty_sd', 'dbty_ss'),
                 ]

    # initialize dict

    kappa_results = {}

    # kappa loop

    for col_sd, col_ss in col_pairs:
        kappa = calculate_kappa(d, col_sd, col_ss)
        kappa_results[f'{col_sd} and {col_ss}'] = kappa

    for pair, kappa in kappa_results.items():
        print(f"Cohen's Kappa for {pair}: {kappa:.2f}")

    # dummy code disagreements Fx

    def encode_disagreements(row):
        return 1 if row[0] != row[1] else 0

    col_dis = [
               ('asp_sd', 'asp_ss', 'asp_dis'),
               ('dep_sd', 'dep_ss', 'dep_dis'),
               ('val_sd', 'val_ss', 'val_dis'),
               #('prg_sd', 'prg_ss', 'prg_dis'),
               #('tgd_sd', 'tgd_ss', 'tgd_dis'),
               #('age_sd', 'age_ss', 'age_dis'),
               #('race_sd', 'race_ss', 'race_dis'),
               #('dbty_sd', 'dbty_ss', 'dbty_dis'),
               ]

    for col1, col2, dis_col in col_dis:
        d[dis_col] = d[[col1, col2]].apply(encode_disagreements, axis = 1)

    # export: cycle-specific

    d.to_excel(f'd_cycle{cycle_num}_iaa.xlsx')

    return d, kappa_results
