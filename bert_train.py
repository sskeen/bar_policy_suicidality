
import torch
import numpy as np
import random

def set_seed(seed):
    """
    Set random seeds for reproducibility in Pytorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
                          BertForSequenceClassification,
                          RobertaForSequenceClassification,
                          DistilBertForSequenceClassification,
                          BertTokenizer,
                          RobertaTokenizer,
                          DistilBertTokenizer,
                          get_linear_schedule_with_warmup,
                          )
from sklearn.metrics import f1_score, matthews_corrcoef, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.nn import CrossEntropyLoss
import numpy as np
import pandas as pd

def train_eval_save_bl_models(target_datasets, targets_and_class_weights, models, save_path, cycle, hyperparameter_grid):
    """
    Fine-tune and eval pre-trained baseline LMs on multiple targets using target-specific train and test datasets.

    Training sets d_train_{target} are split using 5-fold cross-validation: 4 folds for training and 1 fold for validation. Training
    folds use augmented data; validation folds use original data. The best model is selected based on average performance across the
    folds and evaluated on separate test sets d_test_{target}.

    Parameters:
    -----------
    target_datasets : dict
        A dictionary where keys are target names and values are tuples containing the train and test datasets for each target.
        For example: {'asp': (d_train_asp, d_test_asp), 'dep': (d_train_dep, d_test_dep)}

    targets_and_class_weights : dict
        A dictionary where keys are target names and values are lists of class weights corresponding to each target.

    models : dict
        A dictionary of models to evaluate. Each key is a model name, and each value is a tuple containing:
        - the model class,
        - the tokenizer class,
        - the name of the pre-trained model.

    save_path : str, optional
        The path where the best models will be saved.

    cycle : str
        The cycle identifier: 'baseline', indicating performance with prespecified default params; 'adapted', indicating performance
        with in-domain adapted params.

    hyperparameter_grid : dict
        Dictionary containing hyperparameter values: batch_size, gradient_accumulation_steps, learning_rate, num_epochs, warmup_steps, and weight_decay.

    Returns:
    --------
    d_{cycle}_performance : pd.DataFrame
        A df containing performance metrics for each target and model per fold per pre-specified cycle,
        and final evaluation on test data.
    """

    # verify cycle

    print(f"CYCLE: {cycle}")

    # check CUDA

    print("CUDA: ", torch.cuda.is_available())
    use_cuda = torch.cuda.is_available()

    # set seed

    set_seed(56)

    # unpack hyperparameters

    batch_size = hyperparameter_grid.get('batch_size', 4)
    gradient_accumulation_steps = hyperparameter_grid.get('gradient_accumulation_steps', 1)
    learning_rate = hyperparameter_grid.get('learning_rate', 2e-5)
    num_epochs = hyperparameter_grid.get('num_epochs', 2)
    warmup_steps = hyperparameter_grid.get('warmup_steps', 0)
    weight_decay = hyperparameter_grid.get('weight_decay', 0.0)

    # best target x model F1 tracking

    best_f1_scores = {target: {'score': 0, 'model': None, 'model_instance': None} for target in targets_and_class_weights}
    results = []

    # training loop: target x model

    for target, class_weights in targets_and_class_weights.items():
        print("\n======================================================================================")
        print(f"Label: {target}")
        print("======================================================================================")

        # target-specific datasets

        d_train, d_test = target_datasets[target]

        # split augmented v. non-augmented data

        d_train_aug = d_train[d_train['aug'] == 1]
        d_train_no_aug = d_train[d_train['aug'] == 0]

        # prepare fold-wise training v. validation data

        X_train_aug, y_train_aug = d_train_aug['text'].values, d_train_aug[target].values
        X_train_no_aug, y_train_no_aug = d_train_no_aug['text'].values, d_train_no_aug[target].values
        X_test, y_test = d_test['text'].values, d_test[target].values

        # determine target type, encode (as needed)

        target_type = 'binary' if len(np.unique(y_train_aug)) <= 2 else 'multiclass'
        le = LabelEncoder() # Using separate LabelEncoder for each target data group to avoid encoding mismatch issues.
        if target_type == 'binary':
            #le = LabelEncoder()
            y_train_aug = le.fit_transform(y_train_aug)
            y_train_no_aug = le.fit_transform(y_train_no_aug) # Re-encode with new encoder
            y_test = le.transform(y_test)

        # define k folds

        k_fold = StratifiedKFold(
                                 n_splits = 5,
                                 shuffle = True,
                                 random_state = 56,
                                 )

        for model_name, (model_class, tokenizer_class, pretrained_model_name) in models.items():
            print(f"\nFine-tuning {model_name} for {target}")
            print("--------------------------------------------------------------------------------------")

            fold_f1, fold_mcc, fold_auprc = [], [], []  ### store fold-wise performance metrics

            # initialize tokenizer

            tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)

            for fold_idx, (train_no_aug_idx, valid_idx) in enumerate(k_fold.split(X_train_no_aug, y_train_no_aug)):
                print(f"\nFold {fold_idx + 1}/5")

                # create training set: combine aug = 1 (augmented) with fold-specific aug = 0 (non-augmented)

                X_train_fold_aug = X_train_aug
                y_train_fold_aug = y_train_aug

                X_train_fold_no_aug, X_valid_fold = X_train_no_aug[train_no_aug_idx], X_train_no_aug[valid_idx]
                y_train_fold_no_aug, y_valid_fold = y_train_no_aug[train_no_aug_idx], y_train_no_aug[valid_idx]

                # combine augmented and non-augmented training data

                X_train_fold = np.concatenate([X_train_fold_aug, X_train_fold_no_aug])
                y_train_fold = np.concatenate([y_train_fold_aug, y_train_fold_no_aug])

                # tokenize training and validation data

                encoded_train = tokenizer(
                                          X_train_fold.tolist(),
                                          padding = True,
                                          truncation = True,
                                          return_tensors = 'pt',
                                          )

                encoded_valid = tokenizer(
                                          X_valid_fold.tolist(),
                                          padding = True,
                                          truncation = True,
                                          return_tensors = 'pt',
                                          )

                train_dataset = TensorDataset(
                                              encoded_train['input_ids'],
                                              encoded_train['attention_mask'],
                                              torch.tensor(y_train_fold),
                                              )

                valid_dataset = TensorDataset(
                                              encoded_valid['input_ids'],
                                              encoded_valid['attention_mask'],
                                              torch.tensor(y_valid_fold),
                                              )

                train_loader = DataLoader(
                                          train_dataset,
                                          batch_size = batch_size,
                                          shuffle = True,
                                          )

                valid_loader = DataLoader(
                                          valid_dataset,
                                          batch_size = batch_size,
                                          shuffle = False,
                                          )

                # instantiate model

                model = model_class.from_pretrained(pretrained_model_name)

                # migrate to CUDA

                use_cuda = torch.cuda.is_available()
                if use_cuda:
                    model = model.cuda()

                # set optimizer + scheduler

                optimizer = torch.optim.AdamW(
                                              model.parameters(),
                                              lr = learning_rate,
                                              weight_decay = weight_decay,
                                              )

                total_steps = len(train_loader) * num_epochs
                #scheduler = torch.optim.lr_scheduler.LinearLR(
                #                                              optimizer,
                #                                              start_factor = 0.1,
                #                                              #total_iters = warmup_steps,
                #                                              total_iters = total_steps, # Fix: corrected from warmup_steps to total_steps
                #                                              )

                scheduler = get_linear_schedule_with_warmup(
                                                            optimizer,
                                                            num_warmup_steps = warmup_steps,
                                                            num_training_steps = total_steps
                                                            )

                # fine-tune model on training folds (x4)

                model.train()
                criterion = CrossEntropyLoss(weight = torch.tensor(
                                                                   class_weights,
                                                                   dtype = torch.float
                                                                   ).cuda() if use_cuda else torch.tensor(
                                                                                                          class_weights,
                                                                                                          dtype = torch.float
                                                                                                          )
                                             )

                for epoch in range(num_epochs):
                    for i, batch in enumerate(train_loader):
                        input_ids, attention_mask, labels = batch
                        labels = labels.long()

                        if use_cuda:
                            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

                        outputs = model(input_ids, attention_mask = attention_mask)
                        logits = outputs.logits
                        loss = criterion(logits, labels)

                        # accumulate gradients, normalize loss

                        loss = loss / gradient_accumulation_steps
                        loss.backward()

                        # update model weights post-accumulation steps

                        if (i + 1) % gradient_accumulation_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                        # apply learning rate scheduler

                        scheduler.step()

                # evaluate on validation fold (x1)

                model.eval()
                all_predictions, all_true_labels = [], []
                with torch.no_grad():
                    for batch in valid_loader:
                        input_ids, attention_mask, labels = batch

                        if use_cuda:
                            input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

                        outputs = model(input_ids, attention_mask = attention_mask)
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim = 1).tolist()
                        all_predictions.extend(predictions)
                        all_true_labels.extend(labels.tolist())

                # performance metrics per validation fold

                f1_macro = f1_score(
                                    all_true_labels,
                                    all_predictions,
                                    average = 'macro',
                                    )

                mcc = matthews_corrcoef(
                                        all_true_labels,
                                        all_predictions,
                                        )

                auprc = average_precision_score(
                                                all_true_labels,
                                                all_predictions,
                                                average = 'macro',
                                                )

                fold_f1.append(f1_macro)
                fold_mcc.append(mcc)
                fold_auprc.append(auprc)

            # mean results over folds, track best model

            mean_f1 = np.mean(fold_f1)
            if mean_f1 > best_f1_scores[target]['score']:
                best_f1_scores[target]['score'] = mean_f1
                best_f1_scores[target]['model'] = model_name
                best_f1_scores[target]['model_instance'] = model

                save_model_name = f'{target}_{model_name}_best_{cycle}_model.pt'
                torch.save(model.state_dict(), save_path + save_model_name)

            # store results for each fold

            for i in range(5):
                results.append({
                                'target': target,
                                'model': model_name,
                                'fold': i + 1,
                                'f1_macro': fold_f1[i],
                                'mcc': fold_mcc[i],
                                'auprc': fold_auprc[i]
                                })

        # test on held-out d_test_{target} df

        print(f"\nTest on held-out d_test_{target} using the best {best_f1_scores[target]['model']} model")
        print("--------------------------------------------------------------------------------------")

        test_model = best_f1_scores[target]['model_instance']
        test_model.eval()

        #tokenizer = tokenizer_class.from_pretrained(pretrained_model_name)
        #tokenizer = models[best_f1_scores[target]['model']][1].from_pretrained(pretrained_model_name)

        # ensure correct tokenizer for testing

        best_model_name = best_f1_scores[target]['model']  # Retrieve the name of the best model
        best_pretrained_model_name = models[best_model_name][2]  # Retrieve the correct pretrained model name
        tokenizer = models[best_model_name][1].from_pretrained(best_pretrained_model_name)  # Use correct tokenizer class

        encoded_test = tokenizer(
                                 X_test.tolist(),
                                 padding = True,
                                 truncation = True,
                                 return_tensors = 'pt',
                                 )

        test_dataset = TensorDataset(
                                     encoded_test['input_ids'],
                                     encoded_test['attention_mask'],
                                     torch.tensor(y_test),
                                     )

        test_loader = DataLoader(
                                 test_dataset,
                                 batch_size = batch_size,
                                 shuffle = False,
                                 )

        all_test_predictions, all_test_true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch

                if use_cuda:
                    input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()

                outputs = test_model(input_ids, attention_mask = attention_mask)
                logits = outputs.logits
                test_predictions = torch.argmax(logits, dim = 1).tolist()
                all_test_predictions.extend(test_predictions)
                all_test_true_labels.extend(labels.tolist())

        # preformance metrics for held-out d_test_{target} df

        test_f1_macro = f1_score(
                                 all_test_true_labels,
                                 all_test_predictions,
                                 average='macro',
                                 )

        test_mcc = matthews_corrcoef(
                                     all_test_true_labels,
                                     all_test_predictions,
                                     )

        test_auprc = average_precision_score(
                                             all_test_true_labels,
                                             all_test_predictions,
                                             average = 'macro',
                                             )

        # display

        print(f"Test F1 (macro) for {target}: {test_f1_macro}")
        print(f"Test MCC for {target}: {test_mcc}")
        print(f"Test AUPRC for {target}: {test_auprc}")

        # store

        results.append({
                        'target': target,
                        'model': best_f1_scores[target]['model'],
                        'fold': 'Test',
                        'f1_macro': test_f1_macro,
                        'mcc': test_mcc,
                        'auprc': test_auprc
                        })

    # summarize + return d_{cycle}_performance df

    print("\n--------------------------------------------------------------------------------------")
    print(f"Summary")
    print("--------------------------------------------------------------------------------------")

    for target, info in best_f1_scores.items():
        print(f"Best F1 (macro) for {target}: {info['score']} achieved by {info['model']}")

    d_performance = pd.DataFrame(results)
    print(f"\nd_{cycle}_performance:")
    print(d_performance.head(5))
    d_performance.to_excel(f'{save_path}d_{cycle}_performance.xlsx')

from brokenaxes import brokenaxes
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def performance_scatterplot(df, plot_name):
    """
    Creates a categorical scatterplot with custom aesthetics, markers, error bars, and a legend.

    Parameters:
    df (pd.DataFrame): Input dataframe containing columns 'target', 'f1_macro', 'model', and 'fold'.

    plot_name (str): The name used for saving the output plot file (without extension).

    Returns:
    --------
    Matplotlib Axes object containing the barplot.
    """

    # aesthetics

    model_colors = [
                    '#87bc45',
                    '#b33dc6',
                    '#27aeef',
                   ]

      ### SJS 10/1: last three colors in "Retro Metro (Default)" https://www.heavy.ai/blog/12-color-palettes-for-telling-better-stories-with-your-data

    sns.set_style(
                  style = 'whitegrid',
                  rc = None,
                  )

    # map target: numeric position

    target_mapping = {
                      'asp': 0,
                      'dep': 2,
                      'val': 4,
                      'prg': 6,
                      'tgd': 8,
                      'age': 10,
                      'race': 12,
                      'dbty': 14
                     }

    df['target_numeric'] = df['target'].map(target_mapping)
    df['target_numeric'] = pd.to_numeric(df['target_numeric'])

    # inject noise for jitter

    df['target_jitter'] = df['target_numeric'] + np.random.uniform(
                                                                   -0.35,
                                                                   0.35,
                                                                   size = len(df),
                                                                   )

    # initialize fig. with broken y-axis

    fig = plt.figure(figsize=(12, 5.5))
    bax = brokenaxes(
                     ylims = ((0, 0.1), (0.4, 1)), ### y-axis bounds
                     hspace = 0.1, ### y-axis break space
                     )

    # define colors: held-out test set ('fold' = Test)

    test_colors = {
                   'bert': '#87bc45',
                   'roberta': '#27aeef',
                   'distilbert': '#b33dc6',
                   }

    # distinguish markers: fold v. held-out test set

    for fold_value, marker in [('Test', 'o'), ('non-Test', '.')]:
        if fold_value == 'Test':
            data_subset = d_v[d_v['fold'] == 'Test']

            for model in data_subset['model'].unique():
                model_data = data_subset[data_subset['model'] == model]
                bax.scatter(
                            model_data['target_jitter'],
                            model_data['f1_macro'],
                            color = test_colors[model],
                            s = 40,
                            alpha = 0.6,
                            label = None,
                            marker = marker,
                            )
        else:
            data_subset = d_v[d_v['fold'] != 'Test']
            for model, color in test_colors.items():
                model_data = data_subset[data_subset['model'] == model]
                bax.scatter(
                            model_data['target_jitter'],
                            model_data['f1_macro'],
                            color = color,
                            s = 40,
                            alpha = 0.6,
                            label = None,
                            marker = marker,
                            )

#    for fold_value, marker in [('Test', 'o'), ('non-Test', '.')]:
#        if fold_value == 'Test':
#            data_subset = df[df['fold'] == 'Test']
#        else:
#            data_subset = df[df['fold'] != 'Test']

#        sns.scatterplot(
#                        data = data_subset,
#                        x = 'target_jitter',
#                        y = 'f1_macro',
#                        hue = 'model',
#                        palette = model_colors,
#                        s = 40,
#                        alpha = 0.6,
#                        marker = marker,
#                       )

    # mean and SD of f1_macro for each target x model

    mean_std_df = df.groupby(['target', 'model']).agg(
                                                      mean_f1_macro = ('f1_macro', 'mean'),
                                                      std_f1_macro = ('f1_macro', 'std'),
                                                      ).reset_index()

    # add target_numeric values to mean_std_df for plotting means and error bars

    #mean_std_df['target_numeric'] = mean_std_df['target'].map(target_mapping).astype(float)

    # x-axis offsets

    mean_std_df['target_numeric'] = mean_std_df['target'].map(target_mapping).astype(float)
    mean_std_df['target_offset'] = mean_std_df['target_numeric'] + mean_std_df['model'].map(
                                                                                            {'bert': -0.3,
                                                                                             'roberta': 0.0,
                                                                                             'distilbert': 0.3}
                                                                                            )

    #model_offsets = {
    #                 'bert-base-uncased': -0.3,
    #                 'roberta-base': 0.0,
    #                 'distilbert-base-uncased': 0.3,
    #                 }

    #mean_std_df['target_offset'] = mean_std_df['target_numeric'] + mean_std_df['model'].map(model_offsets)

    # means (SDs), error bars

    for model in mean_std_df['model'].unique():
        model_data = mean_std_df[mean_std_df['model'] == model]

    # inspect for NaNs

        if not model_data[['target_offset', 'mean_f1_macro', 'std_f1_macro']].isnull().any().any():
            plt.errorbar(
                         model_data['target_offset'],
                         model_data['mean_f1_macro'],
                         yerr = model_data['std_f1_macro'],
                         fmt = 'D',
                         markersize = 7,
                         capsize = 0,
                         elinewidth = 1,
                         markeredgewidth = 1,
                         color = test_colors[model]
                        )

    # x-tick: map to targets

    bax.set_xlabel(
                   'Target',
                   fontsize = 12,
                   labelpad = 30,
                   )

    bax.set_ylabel(
                   f'$F_1$ (macro): {plot_name}',
                   fontsize = 12,
                   labelpad = 30,
                   )

    # x-tick: label lower axis

    bax.axs[1].set_xticks(list(target_mapping.values()))
    bax.axs[1].set_xticklabels(list(target_mapping.keys()), rotation = 45, fontsize = 10)

    #sns.despine(left = True)
    bax.grid(
             #axis='x',
             False,
             )

    # line at 0.8 threshold

    bax.axhline(
                y = 0.8,
                color = 'r',
                linewidth = 0.6,
                linestyle = '--',
                )

    #plt.xticks(
    #           [0, 2, 4, 6, 8, 10, 12, 14],
    #           ['asp', 'dep', 'val', 'prg', 'tgd', 'age', 'race', 'dbty']
    #          )

    # label axes

    #plt.ylim(0, 1)
    #ax = plt.gca()
    #ax.set_ylabel(
    #              '$F_1$ (macro)',
    #              fontsize = 12,
    #              labelpad = 10,
    #              )

    #ax.set_xlabel(
    #              'Target',
    #              fontsize = 12,
    #              labelpad = 10,
    #              )

    #sns.despine(left = True)
    #ax.grid(axis = 'x')

    # set line at 0.9 threshold

    #ax.axhline(
    #           y = 0.9,
    #           color = 'r',
    #           linewidth = 0.6,
    #           linestyle = '--',
    #           )

    # custom legend

    legend_elements = [
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'bert', markersize = 8, markerfacecolor = '#87bc45', lw = 0),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'roberta', markersize = 8, markerfacecolor = '#27aeef', lw = 0),
                       Line2D([0], [0], marker = 'o', color = 'w', label = 'distilbert', markersize = 8, markerfacecolor = '#b33dc6', lw = 0),
                      ]

    bax.axs[0].legend(
                      handles = legend_elements,
                      loc = 'upper center',
                      bbox_to_anchor = (0.5, 1.15),
                      ncol = 4,
                      fontsize = 9,
                      frameon = False,
                      )

    # save

    file_name = f'{plot_name}_scatter.png'
    plt.savefig(file_name)

    # display

    plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split

def iterative_stratified_train_test_split_with_rationales(df, targets, test_size, random_state):
    """
    Splits df into target-stratified train and test sets for each target in targets list:
    d_train_{target}, d_test_{target}, respectively. Partitions 'rationales' (aug = 1) to train
    set. Returns a dict with target names as keys
    """

    # initialize dict

    target_datasets = {}

    for target in targets:

        # create 'targets' col for stratification

        df_target = df.copy()
        df_target['targets'] = df[target]

        # split augmented vs. non-augmented rows

        aug_rows = df_target[df_target['aug'] == 1]
        non_aug_rows = df_target[df_target['aug'] != 1]

        if non_aug_rows.empty:
            print(f"No non-augmented rows for target {target}. Skipping...")
            continue

        # stratified train-test split on non-augmented rows only

        train_non_aug, test_non_aug = train_test_split(
                                                       non_aug_rows,
                                                       test_size = test_size,
                                                       stratify = non_aug_rows['targets'],
                                                       random_state = random_state,
                                                      )

        # concat augmented rows back into train set

        d_train = pd.concat([train_non_aug, aug_rows])

        # shuffle + reset index: train set

        d_train = d_train.sample(
                                 frac = 1,
                                 random_state = random_state,
                                 ).reset_index(drop = True)

        # retain 'text', 'aug', target cols

        d_train = d_train[['text', 'aug', target]]
        d_test = test_non_aug[['text', 'aug', target]]

        # reset index: test set

        d_test = d_test.reset_index(drop = True)

        # add train and test sets as tuples to target_datasets dict

        target_datasets[target] = (d_train, d_test)

        # inspect

        print(f"\nVerify: d_train_{target} 'aug' count")
        print(d_train['aug'].value_counts(normalize = False))
        print(f"\nVerify: d_test_{target} 'aug' count")
        print(d_test['aug'].value_counts(normalize = False))

        print(f"\n--------------------------------------------------------------------------------------")
        print(f"d_train_{target}: Augmented training data for target '{target}'")
        print(f"--------------------------------------------------------------------------------------")
        print(d_train.shape)
        print(d_train[target].value_counts(normalize = True))
        print(d_train.head(6))

        print(f"\n--------------------------------------------------------------------------------------")
        print(f"d_test_{target}: De-augmented testing data for target '{target}'")
        print(f"--------------------------------------------------------------------------------------")
        print(d_test.shape)
        print(d_test[target].value_counts(normalize = True))
        print(d_test.head(6))

    return target_datasets

import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import CrossEntropyLoss
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
import pandas as pd

def tune_and_optimize_model_hyperparams(tokenizer, model_class, pretrained_model_name, d_train, d_test, target, class_weights, save_path, hyperparameter_grid):
    """
    Tune and optimize model hyperparameters for a specific model-target combination.

    Parameters:
    -----------

    tokenizer:
        Pre-trained tokenizer.

    model_class:
        Pre-trained model class.

    pretrained_model_name:
        Name of the pre-trained model.

    d_train : pd.DataFrame
        Training dataset.

    d_test : pd.DataFrame
        Test dataset.

    target : str
        The target variable for classification.

    class_weights : torch.tensor
        Weights for each class.

    save_path : str
        Path to save the best model.

    hyperparameter_grid : dict
        Dictionary where keys are hyperparameter names and values are lists of possible values.

    Returns:
    --------
    d_test : pd.DataFrame
        Test dataset with predictions and probabilities.

    d_tuned_performance : pd.DataFrame
        DataFrame with the performance metrics for each hyperparameter configuration.
    """

    # check CUDA

    use_cuda = torch.cuda.is_available()
    print("CUDA: ", use_cuda)

    # set seed

    set_seed(56)

    print("======================================================================================")
    print(f"Optimizing: {pretrained_model_name}\nTarget: {target}")
    print("======================================================================================")

    # tokenize train and test sets

    encoded_train = tokenizer(
                              d_train['text'].tolist(),
                              padding = True,
                              truncation = True,
                              return_tensors = 'pt',
                              )

    encoded_test = tokenizer(
                             d_test['text'].tolist(),
                             padding = True,
                             truncation = True,
                             return_tensors = 'pt',
                             )


    # accept dynamic target variables

    train_labels = torch.tensor(d_train[target].values)
    test_labels = torch.tensor(d_test[target].values)

    # prepare datasets

    train_dataset = TensorDataset(
                                  encoded_train['input_ids'],
                                  encoded_train['attention_mask'],
                                  train_labels,
                                  )

    test_dataset = TensorDataset(
                                 encoded_test['input_ids'],
                                 encoded_test['attention_mask'],
                                 test_labels,
                                 )

    #train_loader = DataLoader(
    #                          train_dataset,
    #                          batch_size = 8,  ### to be updated within grid search
    #                          shuffle = True,
    #                          )

    #test_loader = DataLoader(
    #                         test_dataset,
    #                         batch_size = 8,  ### to be updated within grid search
    #                         shuffle = False,
    #                         )

    # initialize class weights

    if use_cuda:
        class_weights = class_weights.cuda()

    # initialize tracking variables

    best_f1_macro = 0
    best_params = None
    best_model_state = None
    best_predictions = []
    best_probabilities = []

    f1_scores = []
    performance_data = []

    # hyperparam grid search: ParameterGrid

    for hyperparams in ParameterGrid(hyperparameter_grid):
        print(f"\nOptimizing with hyperparameters: {hyperparams}")

        train_loader = DataLoader(
                                  train_dataset,
                                  batch_size = hyperparams['batch_size'],
                                  shuffle = True
                                  )
        test_loader = DataLoader(
                                 test_dataset,
                                 batch_size = hyperparams['batch_size'],
                                 shuffle = False
                                 )

        print(f"\nTotal training rows: {len(train_dataset)}")
        print(f"Total evaluation rows: {len(test_dataset)}")
        print(f"Training batch size: {hyperparams['batch_size']}")
        print(f"Evaluation batch size: {hyperparams['batch_size']}")
        print(f"Total training batches: {len(train_loader)}")
        print(f"Total evaluation batches: {len(test_loader)}")
        print("\n")

        # initialize model

        model = model_class.from_pretrained(pretrained_model_name)
        if use_cuda:
            model.cuda()

        # initialize optimizer and lr scheduler

        optimizer = torch.optim.AdamW(
                                      model.parameters(),
                                      lr = hyperparams['learning_rate'],
                                      weight_decay = hyperparams['weight_decay']
                                      )

        # calculate total steps

        total_steps = len(train_loader) * hyperparams['num_epochs']

        # add scheduler with warmup steps

        scheduler = get_linear_schedule_with_warmup(
                                                    optimizer,
                                                    num_warmup_steps = hyperparams['warmup_steps'],
                                                    num_training_steps=total_steps
                                                    )

        criterion = CrossEntropyLoss(weight = class_weights)

        # training loop

        for epoch in range(hyperparams['num_epochs']):
            model.train()
            optimizer.zero_grad()
            for i, batch in enumerate(tqdm(train_loader, desc = f"Training Epoch {epoch + 1}/{hyperparams['num_epochs']}", leave=True)):
                input_ids, attention_mask, labels = batch
                if use_cuda:
                    input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / hyperparams['gradient_accumulation_steps']
                loss.backward()
                if (i + 1) % hyperparams['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()  ### update learning rate
                    optimizer.zero_grad()

        # eval loop

        model.eval()
        all_predictions = []
        all_true_labels = []
        all_probabilities = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = batch
                if use_cuda:
                    input_ids, attention_mask, labels = input_ids.cuda(), attention_mask.cuda(), labels.cuda()
                outputs = model(input_ids, attention_mask=attention_mask)
                probabilities = torch.softmax(outputs.logits, dim = 1)
                predictions = torch.argmax(probabilities, dim = 1).cpu().tolist()
                all_predictions.extend(predictions)
                all_true_labels.extend(labels.cpu().tolist())
                all_probabilities.extend(probabilities.cpu().tolist())

        # calculate F1 (macro)

        current_f1_macro = f1_score(all_true_labels, all_predictions, average='macro')
        f1_scores.append(current_f1_macro)
        print(f"\nCurrent F1 macro with params {hyperparams}: {current_f1_macro}")

        # append F1 and current performance data

        performance_data.append({
                                 'pretrained_model_name': pretrained_model_name,
                                 'target': target,
                                 'f1_score': current_f1_macro,
                                 'batch_size': hyperparams['batch_size'],
                                 'weight_decay': hyperparams['weight_decay'],
                                 'learning_rate': hyperparams['learning_rate'],
                                 'warmup_steps': hyperparams['warmup_steps'],
                                 'num_epochs': hyperparams['num_epochs'],
                                 'gradient_accumulation_steps': hyperparams['gradient_accumulation_steps'],
        })

        if current_f1_macro > best_f1_macro:
            best_f1_macro = current_f1_macro
            best_params = hyperparams
            best_model_state = model.state_dict()
            best_predictions = all_predictions
            best_probabilities = all_probabilities

    #if len(best_predictions) == len(d_test):
    #    d_test['predicted_labels'] = best_predictions
    #    d_test['predicted_probabilities'] = best_probabilities
    #else:
    #    print("Error: Length of predictions does not match length of test set")

    d_test['predicted_labels'] = best_predictions
    d_test['predicted_probabilities'] = best_probabilities

    # save d_test_{target} with pred and prob

    print("--------------------------------------------------------------------------------------")
    print(f"Summary: {target}")
    print("--------------------------------------------------------------------------------------")

    print(d_test.head(6))
    d_test.to_excel(f'{save_path}/d_test_tuned_preds_{target}.xlsx')

    if best_model_state:
        model_path = f"{save_path}/{target}_{pretrained_model_name}_best_tuned_model.bin"
        torch.save(best_model_state, model_path)
        print("\nBest model saved with F1 macro:", best_f1_macro)
        print("Best hyperparameters:", best_params)

    # display F1 scores

    f1_mean = sum(f1_scores) / len(f1_scores)
    f1_std = (sum((x - f1_mean) ** 2 for x in f1_scores) / len(f1_scores)) ** 0.5
    print(f"Mean F1 macro: {f1_mean}")
    print(f"Standard deviation of F1 macro: {f1_std}")

    # df: target-wise

    d_tuned_performance = pd.DataFrame(performance_data)
    print(d_tuned_performance.head(10))

    # save: target-wise df

    d_tuned_performance.to_excel(f'{save_path}/d_tuned_performance_{target}.xlsx')

    return d_test, d_tuned_performance
