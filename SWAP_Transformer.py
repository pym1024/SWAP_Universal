import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy import stats
from src.utils.utilities import *
from src.metrics.swap import SWAP
from src.utils.configuration_electra import ElectraConfig
from src.utils.modeling_electra import ElectraModel
from src.utils.modeling_electra import ElectraLayer
from transformers import ElectraTokenizerFast, DataCollatorWithPadding
from datasets import load_dataset

# Settings for console outputs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser()

# general setting
parser.add_argument('--data_path', default="datasets", type=str, nargs='?', help='path to the image dataset (datasets or datasets/ILSVRC/Data/CLS-LOC)')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--device', default="mps", type=str, nargs='?', help='setup device (cpu, mps or cuda)')
parser.add_argument('--repeats', default=32, type=int, nargs='?', help='times of calculating the training-free metric')
parser.add_argument('--input_samples', default=16, type=int, nargs='?', help='input batch size for training-free metric')

args = parser.parse_args('')

tasks = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst', 'sts', 'glue']

device = torch.device(args.device)

configs = []
with open("datasets/BERT_benchmark.json", 'r') as f:
    configs = json.load(f)

dataset = load_dataset("openwebtext") # Optional: ("glue", "cola") etc.


tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

def tokenize(examples):
    return tokenizer(examples['text'], truncation=True, max_length=512)


tokenized_dataset = dataset.map(tokenize, batched=True, num_proc=32)
tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

dataloader = torch.utils.data.DataLoader(tokenized_dataset['train'], batch_size=128, collate_fn=data_collator)

swap_results = []

for i in tqdm(range(500)):
    # inputs = tokenizer(next(iter(dataloader))['text'], truncation=True, padding='max_length', return_tensors="pt")
    inputs = next(iter(dataloader))
    nas_config = configs[i]["hparams"]["model_hparam_overrides"]["nas_config"]

    config = ElectraConfig(
        nas_config=nas_config, num_hidden_layers=len(nas_config["encoder_layers"]), output_hidden_states=True
    )
    model = ElectraModel(config)
    
    model.to(device)
    inputs.to(device)

    swap = SWAP(model=model, inputs=inputs, device=device, regular=False, mu=0.5, sigma=1.5)
    swap_scores = []
    
    swap.reinit()
    swap_scores.append(swap.forward())

    swap.clear()

    swap_score = np.mean(swap_scores)
    
    swap_results.append([swap_score, configs[i]['scores']['cola'], configs[i]['scores']['mnli'], configs[i]['scores']['mrpc'], configs[i]['scores']['qnli'], configs[i]['scores']['qqp'], configs[i]['scores']['rte'], configs[i]['scores']['sst'], configs[i]['scores']['sts'], configs[i]['scores']['glue']])
    
    
swap_results = pd.DataFrame(swap_results, columns=['swap_score', 'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst', 'sts', 'glue'])

for task in tasks:
    print(f'Spearman Correlation ({task}): {stats.spearmanr(swap_results.swap_score, swap_results[task])[0]}')
    print()
