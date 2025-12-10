import json
import os
from typing import List
import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_token_id = self.tokenizer.token_to_id['<pad>']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['Question']
        answer = item['Answer']
        
        q_ids = self.tokenizer.encode(question, add_special_tokens=True, return_json=False)['token_ids']
        a_ids = self.tokenizer.encode(answer, add_special_tokens=True, return_json=False)['token_ids']
        
        q_ids = q_ids[:self.max_len] + [self.pad_token_id] * (self.max_len - len(q_ids))
        a_ids = a_ids[:self.max_len] + [self.pad_token_id] * (self.max_len - len(a_ids))
        
        return torch.tensor(q_ids, dtype=torch.long), torch.tensor(a_ids, dtype=torch.long)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(src.device)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# This function will load all JSON files from the entire directory
def load_json_data(directory_path):
    all_data = []
    try:
        if not os.path.isdir(directory_path):
            print(f"Error: The directory '{directory_path}' was not found.")
            return None
        
        for filename in os.listdir(directory_path):
            if filename.endswith('.json'):
                file_path = os.path.join(directory_path, filename)
                print(f"Loading data from {file_path}...")
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not isinstance(data, list):
                        raise ValueError(f"JSON file '{filename}' is not a list of objects.")
                    all_data.extend(data)
        
        return all_data

    except FileNotFoundError:
        print(f"Error: The directory '{directory_path}' was not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from a file in '{directory_path}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data: {e}")
        return None