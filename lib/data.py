# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import random
import torch
from datasets import load_dataset
import json
import copy

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc

# Load and process codealpaca dataset
def get_evolcodealpaca(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = json.load(open('/home/abhinav/src/llama-recipes/ft_datasets/evolved_codealpaca_train.json', 'r'))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    indexes = random.sample(range(0, len(traindata)), nsamples)
    for _ in range(nsamples):
    # for index in indexes:

        while True:
            index = random.randint(0, len(traindata) - 1)
            ann = traindata[index]
            prompt = """[Instructions]:\n{instruction}\n\n[Response]:""".format_map(ann)
            trainenc = tokenizer(prompt, return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break

        ann = traindata[index]
        prompt = """[Instructions]:\n{instruction}\n\n[Response]:""".format_map(ann)
       
        example = prompt + ann["output"]
        prompt = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
        example = tokenizer.encode(example)
        example.append(tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = seqlen - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            example = example[: seqlen]
        
        example = example.unsqueeze(0)
        labels = copy.deepcopy(example)
        labels[:, :-1] = -100
        trainloader.append((example, labels))

    return trainloader, trainloader


def get_platypus(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = json.load(open('/home/abhinav/src/llama-recipes/ft_datasets/platypus_openorca1pc_dolphin1pc.json', 'r'))
    PROMPT_DICT = {
        "platypus_prompt_input": (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
        ),
        "platypus_prompt_no_input": (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{instruction}\n\n### Response:"
        ),
        "dolphin_prompt_input": (
            "### System:\n{input}\n\n### User:\n{instruction}\n\n### Response:"
        ),
        "dolphin_prompt_no_input": (
            "### User:\n{instruction}\n\n### Response:"
        ),
        "openorca_prompt_input": (
            "### System:\n{input}\n\n### User:\n{instruction}\n\n### Response:"
        ),
        "openorca_prompt_no_input": (
            "### User:\n{instruction}\n\n### Response:"
        ),
    }

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    indexes = random.sample(range(0, len(traindata)), nsamples)
    for _ in range(nsamples):
    # for index in indexes:

        while True:
            index = random.randint(0, len(traindata) - 1)
            ann = traindata[index]
            if ann.get("input", "") == "":
                prompt = PROMPT_DICT[ann['id'] + "_prompt_no_input"].format_map(ann)
            else:
                prompt = PROMPT_DICT[ann['id'] + "_prompt_input"].format_map(ann)
            trainenc = tokenizer(prompt, return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break

        ann = traindata[index]
        if ann.get("input", "") == "":
            prompt = PROMPT_DICT[ann['id'] + "_prompt_no_input"].format_map(ann)
        else:
            prompt = PROMPT_DICT[ann['id'] + "_prompt_input"].format_map(ann)
       
        example = prompt + ann["output"]
        prompt = torch.tensor(
            tokenizer.encode(prompt), dtype=torch.int64
        )
        example = tokenizer.encode(example)
        example.append(tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        padding = seqlen - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64)))
        elif padding < 0:
            example = example[: seqlen]
        
        example = example.unsqueeze(0)
        labels = copy.deepcopy(example)
        labels[:, :-1] = -100
        trainloader.append((example, labels))

    return trainloader, trainloader

# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    if "codealpaca" in name:
        return get_evolcodealpaca(nsamples, seed, seqlen, tokenizer)
    if "platypus" in name:
        return get_platypus(nsamples, seed, seqlen, tokenizer)