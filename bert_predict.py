
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
                          DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          BertTokenizer,
                          BertForSequenceClassification,
                          RobertaTokenizer,
                          RobertaForSequenceClassification,
                          )
from tqdm import tqdm
import pandas as pd

def load_model(model_path, model_class, pretrained_model_name):
    """
    Loads a pre-trained fine-tined LM from a specified path.
    """
    model = model_class.from_pretrained(pretrained_model_name)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
                          DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          BertTokenizer,
                          BertForSequenceClassification,
                          RobertaTokenizer,
                          RobertaForSequenceClassification,
                          )
from tqdm import tqdm
import pandas as pd

def preprocess_data(tokenizer, texts):
    """
    Tokenizes a list of texts using the specified LM-specific tokenizer.
    """
    encoded_texts = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    return encoded_texts

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
                          DistilBertTokenizer,
                          DistilBertForSequenceClassification,
                          BertTokenizer,
                          BertForSequenceClassification,
                          RobertaTokenizer,
                          RobertaForSequenceClassification,
                          )
from tqdm import tqdm
import pandas as pd

def predict(model, tokenizer, texts, batch_size = 8, use_cuda = True):
    """
    Predicts labels and probabilities for a list of texts using the specified model and tokenizer.
    """
    print(f"Total number of texts to predict: {len(texts)}")
    encoded_texts = preprocess_data(tokenizer, texts)
    dataset = TensorDataset(encoded_texts['input_ids'], encoded_texts['attention_mask'])
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print(f"Batch size: {batch_size}")
    print(f"Total number of batches: {len(data_loader)}")

    if use_cuda:
        model.cuda()

    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        progress_bar = tqdm(total=len(data_loader), desc="Predicting", leave=False)
        for batch in data_loader:
            input_ids, attention_mask = batch
            if use_cuda:
                input_ids, attention_mask = input_ids.cuda(), attention_mask.cuda()

            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_probabilities.extend(probabilities.cpu().tolist())
            progress_bar.update(1)
        progress_bar.close()

    return all_predictions, all_probabilities
