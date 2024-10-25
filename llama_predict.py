
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def llama_load_and_predict_single_target(target, df, models_path, batch_size):
    """
    Function to load a single model and tokenizer for a specified target, and use them to predict
    labels and class probabilities for the text in df['text'] in batches.

    Args:
        target (str): The name of the target (e.g., 'asp', 'dep').
        df (pd.DataFrame): DataFrame containing a 'text' column with the input texts.
        models_path (str): Directory where the model for the target is saved (e.g., /models/).
        batch_size (int): The batch size for processing the data in smaller chunks.

    Returns:
        pd.DataFrame: DataFrame with additional columns '{target}_pred' and '{target}_prob'.
    """

    # load target-specific best-performing tuned Llama

    model_save_path = f'{models_path}/{target}_llama_best_tuned_model'

    tokenizer = AutoTokenizer.from_pretrained(model_save_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_save_path)

    # set padding token

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # migrate to GPU

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ensure eval mode

    model.eval()

    predicted_labels = []
    class_probabilities = []

    # batch processing

    with torch.no_grad():
        for i in range(0, len(df), batch_size):
            batch_texts = df['text'][i:i + batch_size].tolist()

            # tokenize

            inputs = tokenizer(
                               batch_texts,
                               return_tensors = 'pt',
                               truncation = True,
                               padding = True,
                               max_length = 512,
                               )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            # get logits

            outputs = model(**inputs)
            logits = outputs.logits

            # logits -> probabilities via softmax

            probabilities = torch.softmax(logits, dim = -1).cpu().numpy()

            # get predicted labels

            predicted_labels_batch = torch.argmax(logits, dim = -1).cpu().numpy()

            # append results

            predicted_labels.extend(predicted_labels_batch)
            class_probabilities.extend(probabilities)

    # to df

    df[f'{target}_pred'] = predicted_labels
    df[f'{target}_prob'] = class_probabilities

    return df

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

def llama_load_and_predict_multi_target(targets, df, models_path, batch_size):
    """
    Function to load multiple models and tokenizers for multiple targets, and use them to predict
    labels and class probabilities for the text in df['text'] in batches.

    Args:
        targets (list): List of target names (e.g., ['asp', 'dep']).
        df (pd.DataFrame): DataFrame containing a 'text' column with the input texts.
        models_path (str): Directory where models for each target are saved (e.g., /models/).
        batch_size (int): The batch size for processing the data in smaller chunks.

    Returns:
        pd.DataFrame: DataFrame with additional columns '{target}_pred' and '{target}_prob' for each target.
    """

    for target in targets:

        # load target-specific best-performing tuned Llama

        model_save_path = f'{models_path}/{target}_llama_best_tuned_model'

        tokenizer = AutoTokenizer.from_pretrained(model_save_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_save_path)

        # set padding token

        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        # migrate to GPU

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # ensure eval mode

        model.eval()

        predicted_labels = []
        class_probabilities = []

        # batch processing

        with torch.no_grad():
            for i in range(0, len(df), batch_size):
                batch_texts = df['text'][i:i + batch_size].tolist()

                # tokenize

                inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # get logits

                outputs = model(**inputs)
                logits = outputs.logits

                # logits -> probabilities via softmax

                probabilities = torch.softmax(logits, dim = -1).cpu().numpy()

                # get predicted labels

                predicted_labels_batch = torch.argmax(logits, dim = -1).cpu().numpy()

                # append results

                predicted_labels.extend(predicted_labels_batch)
                class_probabilities.extend(probabilities)

        # to df

        df[f'{target}_pred'] = predicted_labels
        df[f'{target}_prob'] = class_probabilities

    return df
