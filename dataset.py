import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import pandas as pd
import os
import os.path as path
import re
import logging
from tqdm.auto import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer

import torch
from torch.utils.data import (DataLoader, TensorDataset,
                              RandomSampler, SequentialSampler, WeightedRandomSampler)

logger = logging.getLogger()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# Model parameter
MAX_SEQ_LEN = 128


def tokenize_sample(text, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(text) 
    tokens = tokens[ : (max_seq_length - 2)] 
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]

    token_len = len(tokens)
    padding_len = max_seq_length - token_len

    tokens = tokens + ([tokenizer.pad_token] * padding_len)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * token_len
    input_mask = input_mask + ([0] * padding_len)

    assert(len(input_ids) == max_seq_length)
    assert(len(input_mask) == max_seq_length)

    return input_ids, input_mask


def tokenize(df, tokenizer, desc):
    input_ids = []
    input_masks = []
    
    # field = "Raw_Tweet" if isinstance(tokenizer, AutoTokenizer) else "Text"
    field = "Text"

    for index, row in tqdm(df.iterrows(), desc=desc, leave=False):
        sent = row[field]
        ids, mask = tokenize_sample(sent, tokenizer, MAX_SEQ_LEN)
        input_ids.append(ids)
        input_masks.append(mask)

    return np.array(input_ids), np.array(input_masks)


def normalize_label(row, label):
    if row[label] == 1:
        return -1.0
    return (row[label] - 1.0) / 6.0


def read_files(config, args, read_ensemble):
    data_home = config.data_home
    if read_ensemble: data_home = config.ensemble_home
    
    train_df = pd.read_csv(path.join(data_home, config.labeled))
    test_df = pd.read_csv(path.join(config.data_home, config.test))
    unlabeled_df = pd.read_csv(path.join(data_home, config.unlabeled), lineterminator="\n")
    unlabeled_df = unlabeled_df.head(config.new_samples)
    
    if args.normalize_label:
        label = args.label
        normlabel = label + "_norm"
        logger.info("Normalizing label {} in the range of [1.0, 7.0]".format(label))
        train_df[normlabel] = train_df.apply(normalize_label, args=(label,), axis=1)
        test_df[normlabel] = test_df.apply(normalize_label, args=(label,), axis=1)
    
    return train_df, test_df, unlabeled_df


def get_dataset(tokenizer, args, config=None, iteration=0):
    label = args.label
    
    # read main data if iteration is 0, otherwise read ensemble data
    train_df, test_df, unlabeled_df = read_files(config, args, iteration > 0)
    
    if args.normalize_label:
        label += "_norm"
    
    train_df, val_df = train_test_split(train_df, test_size=15/85, random_state=47)
    
    # tokenize the text data using Huggingface tokenizer
    X_train, mask_train = tokenize(train_df, tokenizer, "Tokenize Train") 
    y_train = train_df[label].to_numpy() 
    
    X_val, mask_val = tokenize(val_df, tokenizer, "Tokenize Val")
    y_val = val_df[label].to_numpy()
    
    X_test, mask_test = tokenize(test_df, tokenizer, "Tokenize Test")
    y_test = test_df[label].to_numpy()
    
    train_data = TensorDataset(torch.tensor(X_train, dtype=torch.long), 
                                torch.tensor(mask_train, dtype=torch.long), 
                                torch.tensor(y_train, dtype=torch.float))
    
    val_data = TensorDataset(torch.tensor(X_val, dtype=torch.long), 
                                torch.tensor(mask_val, dtype=torch.long), 
                                torch.tensor(y_val, dtype=torch.float))
    
    test_data = TensorDataset(torch.tensor(X_test, dtype=torch.long), 
                                torch.tensor(mask_test, dtype=torch.long), 
                                torch.tensor(y_test, dtype=torch.float))

#     cache_file = os.path.join(config.cache_path, "unlabeled_processed_" + str(iteration) + ".pt")
#     if config.overwrite_cache or not os.path.exists(cache_file):
    X_unlabeled, mask_unlabeled = tokenize(unlabeled_df, tokenizer, "Tokenize unlabeled") 
#         torch.save((X_unlabeled, mask_unlabeled), cache_file)
#     else:
#         print ("Loading processed unlabeled samples from {}".format(cache_file))
#         X_unlabeled, mask_unlabeled = torch.load(cache_file, map_location=device)
    
    unlabeled_data = TensorDataset(torch.tensor(X_unlabeled, dtype=torch.long),
                                    torch.tensor(mask_unlabeled, dtype=torch.long))

    return train_data, val_data, test_data, unlabeled_data



def get_weights(dataset):
    sampler = SequentialSampler(dataset)
    for batch in DataLoader(dataset, sampler=sampler, batch_size=len(dataset)):
        comments, masks, labels = batch
    labels = labels.cpu().numpy()
    
    n_bins = 2
    cnt_bins = [0, 0]
    bin_map = [-1 for _ in range(len(labels))]
    
    for i, label in enumerate(labels):
        binnum = 0 if label < 0 else 1
        bin_map[i] = binnum
        cnt_bins[binnum] += 1
    
#     sns.scatterplot(x=labels, y=list(range(n_labels)))
#     plt.show()
    
    weights_per_bin = [1.0/cnt_bins[b] for b in range(n_bins)]
    weights = [weights_per_bin[bin_map[i]] for i in range(len(labels))]
    
#     sns.histplot(weights, bins=n_bins)
#     plt.show()
    
    return weights


def get_iterator(dataset, batch_size, shuffle=True, weighted=False):
#     print ("Get iterator of dataset with length {}".format(len(dataset)))
    if shuffle:
        if weighted:
            weights = get_weights(dataset)
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataset_iter = DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size
    )
    return dataset_iter


def add_ensemble_data(values, variances, config, label="Label", iteration=0):
    print ("Ensemble iteration", iteration) 
    print ("Total samples", len(values)) 
    
    values = [v*6.0 + 1. for v in values]
    
    data_home = config.data_home
    if iteration>0: 
        data_home = config.ensemble_home
    
    train_df = pd.read_csv(path.join(data_home, config.labeled))
    
    unlabeled_df = pd.read_csv(path.join(data_home, config.unlabeled), lineterminator="\n") 
    newpool_df = unlabeled_df.head(config.new_samples).copy()

    newpool_df.loc[:, label] = values
    newpool_df.loc[:, "ens_variance"] = variances
    
    # filter by ensemble variances
    print ("Filtering samples by ensemble variance")
    
    # factor of max variance to set the limit in selected samples
    # var_factor = 5
    # max_variance = max(variances)
    # variance_limit = max_variance / var_factor
    # print ("Max variance is {} and variance limit is set to {}".format(max_variance, variance_limit))
    
    variances_sorted = variances.copy()
    variances_sorted.sort()
    variance_limit = variances_sorted[train_df.shape[0]*4//5]
    
    certain_df = newpool_df[newpool_df['ens_variance'] <= variance_limit]
    print ("Filtered samples based on ensemble variances are {}".format(certain_df.shape[0]))
    
    uncertain_df = newpool_df[newpool_df['ens_variance'] > variance_limit]
    
    if iteration > 0:
        os.rename(path.join(data_home, config.unlabeled), path.join(data_home, str(iteration-1) + config.unlabeled))
        os.rename(path.join(data_home, config.labeled), path.join(data_home, str(iteration-1) + config.labeled))
    
    train_df = train_df.append(certain_df, sort=True)
    train_df.to_csv(path.join(config.ensemble_home, config.labeled), index=False) 
    
    N = unlabeled_df.shape[0]
    unlabeled_df = unlabeled_df.tail(N - config.new_samples).copy()
    unlabeled_df = unlabeled_df.append(uncertain_df, sort=True)
    unlabeled_df.to_csv(path.join(config.ensemble_home, config.unlabeled), index=False)
    
    return unlabeled_df 

