
import torch
from transformers import(
                         AutoTokenizer,
                         AutoModelForSequenceClassification,
                         BitsAndBytesConfig,
                         )
from peft import(
                 get_peft_model,
                 LoraConfig,
                 prepare_model_for_kbit_training,
                 )

def load_llama_and_tokenizer(model_name, num_labels):
    """
    Loads the Llama model and tokenizer with 4-bit quantization and LoRA (Low-Rank Adaptation) applied.

    Args:
    model_name (str): The name or path of the pretrained Llama model.
    num_labels (int): The number of labels for the classification task (binary or multiclass).

    Returns:
    model (AutoModelForSequenceClassification): the Llama model configured for sequence classification.
    tokenizer (AutoTokenizer): the tokenizer associated with the Llama model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space = True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
                                             load_in_4bit = True,
                                             bnb_4bit_quant_type = 'nf4',
                                             bnb_4bit_use_double_quant = True,
                                             bnb_4bit_compute_dtype = torch.bfloat16,
                                             )

    model_name = model_name

    model = AutoModelForSequenceClassification.from_pretrained(
                                                               model_name,
                                                               quantization_config = quantization_config,
                                                               num_labels = num_labels,
                                                               device_map = 'auto',
                                                               )


    # apply LoRA

    lora_config = LoraConfig(
                             r = 16,
                             lora_alpha = 8,
                             target_modules = [
                                               'q_proj',
                                               'k_proj',
                                               'v_proj',
                                               'o_proj',
                                               ],
                             lora_dropout = 0.05,
                             bias = 'none',
                             task_type = 'SEQ_CLS',
                             )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    return model, tokenizer

from transformers import PreTrainedTokenizer

def llama_tokenize(examples, tokenizer):
    """
    Tokenize the input examples using the provided tokenizer.

    Args:
    examples (dict): a dictionary containing the text to be tokenized. Assumes the key 'text' contains the input text.
    tokenizer (PreTrainedTokenizer): the tokenizer to be used for tokenizing the text.

    Returns:
    dict: a dictionary with tokenized input including input_ids, attention_mask, etc.
    """

    # tokenize 'text' col

    return tokenizer(
                     examples['text'],
                     padding = 'max_length',
                     truncation = True,
                     max_length = 512,
                     )

from sklearn.metrics import average_precision_score
#from datasets import load_metric
import evaluate

# load metrics

f1_metric = evaluate.load('f1')
mcc_metric = evaluate.load('matthews_correlation')

#f1_metric = load_metric('f1')
#mcc_metric = load_metric('matthews_correlation')

def compute_llama_metrics(eval_pred):
    """
    Compute evaluation metrics for the Llama model during evaluation.

    Args:
    eval_pred (tuple): a tuple containing predictions and labels. The predictions are logits, and the labels are the ground truth.

    Returns:
    dict: a dictionary containing F1 (macro), AUPRC, and MCC scores.
    """
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    f1 = f1_metric.compute(predictions = preds, references = labels, average = 'macro')
    auprc = average_precision_score(labels, predictions[:, 1]) ### use second (pos) class for binary classification
    mcc = mcc_metric.compute(predictions=preds, references = labels)

    return {
            'f1_macro': f1,
            'auprc': auprc,
            'mcc': mcc,
            }

from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import ParameterGrid, StratifiedKFold
import torch
from transformers import(
                         #Accelerator,
                         AdamW,
                         TrainingArguments,
                         Trainer,
                         )

def train_and_evaluate_llama(target_datasets, targets_and_class_weights, model_name, hyperparameter_grid, save_path):
    """
    Trains and tests Llama for multiple targets using stratified k-fold cross-validation
    and a held-out test set. The function handles data preparation, model loading, tokenization,
    training with Hugging Face's Trainer, and performance metrics for each target.

    Args:
    target_datasets (dict): A dictionary where keys are target names and values are tuples of
    (d_train_{target}, d_test_{target}) for each target.
    targets_and_class_weights (dict): A dictionary of class weights for each target to mitigate class imbalance.
    model_name (str): The name or path of the pretrained Llama model to load.
    hyperparameter_grid (list): grid space of hyperparameter configurations (generated using ParameterGrid).
    save_path (str): directory to save best-performing model

    Returns:
    Saves best-performing model by target, saves df of tabulated performance metrics.
    """

    # initialize accelerator

    accelerator = Accelerator()

    # HF login

    login(token = '')

    # initialize performance df

    d_llama_performance = pd.DataFrame(columns = [
                                                  'target',
                                                  'model',
                                                  'fold',
                                                  'f1_macro',
                                                  'mcc',
                                                  'auprc',
                                                  ]
                                      )

    for target, (d_train, d_test) in target_datasets.items():
        class_weights = torch.tensor(targets_and_class_weights[target]).to(accelerator.device)
        print("\n======================================================================================")
        print(f"Training Llama for target: {target}")
        print("======================================================================================")

        # prep data for cross-validation

        skf = StratifiedKFold(n_splits = 5)
        aug_mask = d_train['aug'] == 1 ### augmentation mask, aug = 1 in training folds only

        # load model, tokenizer

        model, tokenizer = load_llama_and_tokenizer(model_name, num_labels = 2)

        # train-validation loop

        for fold, (train_index, val_index) in enumerate(skf.split(d_train, d_train[target])):

            print(f"\n")
            print(f"\nFold {fold + 1}/5")

            train_mask = aug_mask | d_train.index.isin(train_index)
            val_mask = ~aug_mask & d_train.index.isin(val_index)

            # split train and validation sets based on aug mask

            d_train_fold = d_train[train_mask].copy()
            d_val_fold = d_train[val_mask].copy()

            print(f"Fold {fold + 1} Training rows: {len(d_train_fold)}")
            print(f"Fold {fold + 1} Validation rows: {len(d_val_fold)}")

            # rename 'target' col to 'label' for HF Trainer

            d_train_fold = d_train_fold.rename(columns = {target: 'label'})
            d_val_fold = d_val_fold.rename(columns = {target: 'label'})

            # excise 'aug' col before creating HF Dataset objects

            d_train_fold = d_train_fold.drop(columns = ['aug'])
            d_val_fold = d_val_fold.drop(columns = ['aug'])

            # convert to HF Dataset

            train_dataset = Dataset.from_pandas(d_train_fold)
            val_dataset = Dataset.from_pandas(d_val_fold)

            # tokenize

            train_dataset = train_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)
            val_dataset = val_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)

            # reformat to PyTorch tensors for HF Trainer compatibility

            train_dataset.set_format(type = 'torch', columns = [
                                                                'input_ids',
                                                                'attention_mask',
                                                                'label',
                                                                ]
                                     )

            val_dataset.set_format(type = 'torch', columns = [
                                                              'input_ids',
                                                              'attention_mask',
                                                              'label',
                                                              ]
                                   )

            # display training and validation details

            train_batch_size = 4 ### mirrors training_args
            val_batch_size = 4
            total_train_batches = len(train_dataset) // train_batch_size
            total_eval_batches = len(val_dataset) // val_batch_size

            print(f"Total training rows: {len(train_dataset)}")
            print(f"Total validation rows: {len(val_dataset)}")
            print(f"Training batch size: {train_batch_size}")
            print(f"Validation batch size: {val_batch_size}")
            print(f"Total training batches: {total_train_batches}")
            print(f"Total evaluation batches: {total_eval_batches}")

            # HF TrainingArguments

            for h in hyperparameter_grid:
                training_args = TrainingArguments(
                                                  output_dir = '/content/drive/MyDrive/Colab/bar_policy_suicidality/temp/',
                                                  learning_rate = h['learning_rate'],
                                                  per_device_train_batch_size = 4, ### RAM overhead: reduced batch size
                                                  per_device_eval_batch_size = 4,
                                                  num_train_epochs = h['num_train_epochs'],
                                                  weight_decay = h['weight_decay'],
                                                  gradient_accumulation_steps = h['gradient_accumulation_steps'],
                                                  warmup_steps = h['warmup_steps'],
                                                  evaluation_strategy = 'epoch',
                                                  save_strategy = 'epoch',
                                                  report_to = 'none',
                                                  push_to_hub = False,
                                                  remove_unused_columns = True, ### 'aug' dropped here
                                                  fp16 = True,  ### RAM overhead: mixed precision enabled
                                                  seed = 56,
                                                  )

            # HF Trainer setup

            trainer = Trainer(
                              model = model,
                              args = training_args,
                              train_dataset = train_dataset,
                              eval_dataset = val_dataset,
                              compute_metrics = compute_llama_metrics,
                              optimizers = (AdamW(model.parameters(), lr = training_args.learning_rate), None),
                              )

            # train

            trainer.train()

            # append fold metrics to performance dataframe

            val_metrics = trainer.evaluate(val_dataset)
            d_llama_performance.loc[len(d_llama_performance)] = [
                target, 'llama-3.1-8b', fold + 1, val_metrics['eval_f1_macro'], val_metrics['eval_mcc'], val_metrics['eval_auprc']
            ]

        # test on held-out test set

        print("--------------------------------------------------------------------------------------")
        print(f"Testing Llama for target: {target}")
        print("--------------------------------------------------------------------------------------")

        # rename 'target' col: held-out test set

        d_test = d_test.rename(columns = {target: 'label'})

        # excise 'aug' col: held-out test set

        d_test = d_test.drop(columns = ['aug'])

        test_dataset = Dataset.from_pandas(d_test)
        test_dataset = test_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)
        test_dataset.set_format(type = 'torch', columns = ['input_ids', 'attention_mask', 'label'])

        # display test set details

        test_batch_size = 4
        total_test_batches = len(test_dataset) // test_batch_size

        print(f"Total test rows: {len(test_dataset)}")
        print(f"Test batch size: {test_batch_size}")
        print(f"Total test batches: {total_test_batches}")

        # test

        test_metrics = trainer.evaluate(test_dataset)
        print(test_metrics)

        d_llama_performance.loc[len(d_llama_performance)] = [
            target, 'llama-3.1-8b', 'Test', test_metrics['eval_f1_macro'], test_metrics['eval_mcc'], test_metrics['eval_auprc']
        ]

        # save target-wise trained models

        print(f"\nSaving benchmark trained Llama for target: {target}")
        target_save_path = f'{save_path}/{target}_llama_benchmark_model'
        model.save_pretrained(target_save_path)
        tokenizer.save_pretrained(target_save_path)

    # extract performance scores numeric values

    d_llama_performance['f1_macro'] = d_llama_performance['f1_macro'].apply(lambda i: i['f1'] if isinstance(i, dict) else i)
    d_llama_performance['mcc'] = d_llama_performance['mcc'].apply(lambda i: i['matthews_correlation'] if isinstance(i, dict) else i)
    d_llama_performance['auprc'] = d_llama_performance['auprc'].apply(lambda i: i if isinstance(i, float) else None)  # Ensure AUPRC is numeric

    print("\n--------------------------------------------------------------------------------------")
    print(f"Summary: Llama performance for target: {target}")
    print("--------------------------------------------------------------------------------------")

    print(d_llama_performance.head(6))
    d_llama_performance.to_excel('d_llama_performance.xlsx')

from accelerate import Accelerator
from datasets import Dataset
from huggingface_hub import login
import pandas as pd
from sklearn.model_selection import ParameterGrid, train_test_split
import torch
from transformers import(
                         #Accelerator,
                         AdamW,
                         TrainingArguments,
                         Trainer,
                         )

def tune_and_optimize_llama_hyperparams(target_datasets, targets_and_class_weights, model_name, hyperparameter_grid, save_path):
    """
    Tune and optimize hyperparameters for a Llama model using ParameterGrid search.
    For each target, trains, validates (8:2 split), and tests on held-out d_test_{target}, adjusting
    model in accord with pre-specified ParameterGrid. Saves best-performing model by target.

    Args:
    target_datasets (dict): A dictionary where keys are target names and values are tuples of
    (d_train_{target}, d_test_{target}) for each target.
    targets_and_class_weights (dict): A dictionary of class weights for each target to mitigate class imbalance.
    model_name (str): The name or path of the pretrained Llama model to load.
    hyperparameter_grid (list): grid space of hyperparameter configurations (generated using ParameterGrid).
    save_path (str): directory to save best-performing model

    Returns:
    Saves best-performing model by target, saves df of tabulated performance metrics.
    """
    # initialize accelerator

    accelerator = Accelerator()

    # HF login

    login(token = '')

    # initialize performance df

    d_llama_performance = pd.DataFrame(columns = [
                                                  'target',
                                                  'model',
                                                  'f1_macro',
                                                  'mcc',
                                                  'auprc',]
                                      )

    for target, (d_train, d_test) in target_datasets.items():
        class_weights = torch.tensor(targets_and_class_weights[target]).to(accelerator.device)
        print("\n======================================================================================")
        print(f"Tuning Llama 3.1 for target: {target}")
        print("======================================================================================")

        best_f1_macro = 0 ### tracking var: best F1 (macro)
        best_model_state = None ### tracking var: best-performing model x hyperparam configs

        # load model, tokenizer

        model, tokenizer = load_llama_and_tokenizer(model_name, num_labels=2)

        for h in hyperparameter_grid:
            print("\n")
            print(f"\nTuning with hyperparam config: {h}")

            # split d_train into a smaller validation set

            d_train, d_val = train_test_split(
                                              d_train,
                                              test_size = 0.2,
                                              stratify = d_train[target],
                                              )



            # rename 'target' col to 'label' for HF Trainer

            d_train = d_train.rename(columns = {target: 'label'})
            d_val = d_val.rename(columns = {target: 'label'})
            d_test = d_test.rename(columns = {target: 'label'})

            # convert to HF Dataset

            train_dataset = Dataset.from_pandas(d_train)
            val_dataset = Dataset.from_pandas(d_val)
            test_dataset = Dataset.from_pandas(d_test)

            # tokenize

            train_dataset = train_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)
            val_dataset = val_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)
            test_dataset = test_dataset.map(lambda i: llama_tokenize(i, tokenizer), batched = True)

            # reformat to PyTorch tensors for HF Trainer compatibility

            train_dataset.set_format(type = 'torch', columns = [
                                                                'input_ids',
                                                                'attention_mask',
                                                                'label',
                                                                ]
                                     )

            val_dataset.set_format(type = 'torch', columns = [
                                                              'input_ids',
                                                              'attention_mask',
                                                              'label',
                                                              ]
                                   )

            test_dataset.set_format(type = 'torch', columns = [
                                                               'input_ids',
                                                               'attention_mask',
                                                               'label',
                                                               ]
                                    )

            # display training and testing details

            train_batch_size = 4
            test_batch_size = 4
            total_train_batches = len(train_dataset) // train_batch_size
            total_test_batches = len(test_dataset) // test_batch_size

            print(f"Total training rows: {len(train_dataset)}")
            print(f"Total validation rows: {len(d_val)}")
            print(f"Total test rows: {len(test_dataset)}")
            print(f"Training batch size: {train_batch_size}")
            print(f"Test batch size: {test_batch_size}")
            print(f"Total training batches: {total_train_batches}")
            print(f"Total test batches: {total_test_batches}")

            # HF TrainingArguments w/ ParameterGrid

            training_args = TrainingArguments(
                                              output_dir = '/content/drive/MyDrive/Colab/bar_policy_suicidality/temp/',
                                              learning_rate = h['learning_rate'],
                                              per_device_train_batch_size = train_batch_size,
                                              per_device_eval_batch_size = test_batch_size,
                                              num_train_epochs = h['num_train_epochs'],
                                              weight_decay = h['weight_decay'],
                                              gradient_accumulation_steps = h['gradient_accumulation_steps'],
                                              warmup_steps = h['warmup_steps'],
                                              evaluation_strategy = 'epoch',
                                              save_strategy = 'epoch',
                                              report_to = 'none',
                                              push_to_hub = False,
                                              remove_unused_columns = True, ### 'aug' dropped here
                                              fp16 = True, ### mixed precision to mitigate memory overhead
                                              seed = 56,
                                              )

            # HF Trainer setup

            trainer = Trainer(
                              model = model,
                              args = training_args,
                              train_dataset = train_dataset,
                              eval_dataset = val_dataset,
                              compute_metrics = compute_llama_metrics,
                              optimizers = (AdamW(model.parameters(), lr = training_args.learning_rate), None),
                              )

            # train

            trainer.train()

            # test on held-out test set

            print("--------------------------------------------------------------------------------------")
            print(f"Testing Llama for target: {target}")
            print("--------------------------------------------------------------------------------------")

            test_metrics = trainer.evaluate(test_dataset)
            print(test_metrics)

            # append fold metrics to performance dataframe

            d_llama_performance.loc[len(d_llama_performance)] = [
                target, 'llama-3.1-8b', test_metrics['eval_f1_macro'], test_metrics['eval_mcc'], test_metrics['eval_auprc']
            ]

            # save best model based on F1 (macro)

            if test_metrics['eval_f1_macro']['f1'] > best_f1_macro:
                best_f1_macro = test_metrics['eval_f1_macro']['f1']
                #best_model_state = model.state_dict() ### save model state
                print(f"\nUpdating best model state for target: {target} with F1 (macro): {best_f1_macro}")

        # save the best model target-wise

        #if best_model_state:
        if best_f1_macro > 0:
            print(f"\nSaving best model for target: {target} with F1 (macro): {best_f1_macro}")

            # load best-performing model state

            #model.load_state_dict(best_model_state)

            # save base quantized model (without LoRA)

            target_save_path = f'{save_path}/{target}_llama_best_tuned_model'
            model.save_pretrained(target_save_path)

            # save LoRA adapter separately
            #adapter_save_path = f'{save_path}/{target}_llama_best_tuned_adapter'
            #model.save_adapter(adapter_save_path)

            # save tokenizer

            tokenizer.save_pretrained(target_save_path)

    # extract performance scores numeric values

    d_llama_performance['f1_macro'] = d_llama_performance['f1_macro'].apply(lambda i: i['f1'] if isinstance(i, dict) else i)
    d_llama_performance['mcc'] = d_llama_performance['mcc'].apply(lambda i: i['matthews_correlation'] if isinstance(i, dict) else i)
    d_llama_performance['auprc'] = d_llama_performance['auprc'].apply(lambda i: i if isinstance(i, float) else None) ### ensure AUPRC is numeric

    print("Llama performance summary:")
    d_llama_performance.to_excel('d_llama_tuned_performance.xlsx')
