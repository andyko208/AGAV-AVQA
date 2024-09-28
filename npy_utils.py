import os
import re
import ast
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel
from torch.utils.data import IterableDataset, Dataset, DataLoader


"""

Replace question with placeholders with templ_values
>>> question_content: 'What is the <Object> making sound?'
>>> templ_values: ['violin']
>>> 'What is the violin making sound?'

"""
def replace_placeholders(question_content, templ_values):
    
    templ_values = ast.literal_eval(templ_values)
    
    replaced_content = re.sub(r'<[^>]*>', lambda x: templ_values.pop(), question_content)
    
    return replaced_content


"""

Returns one-hot representation of the answer(label)
>>> key: 'yes'
>>> vocab_dict: {'yes': 0, 'no': 1, ...}
>>> tensor([1, 0, ...])

"""
def one_hot(key, vocab_dict):

    vocab_len = 42  # len(vocab_dict)
    ind_mapping = torch.tensor(vocab_dict[key])
    encoding = F.one_hot(ind_mapping, vocab_len).to(torch.float32)
    
    return encoding

"""

a_feat[key=video_id]: dictionary of audio features (60, 128)
v_feat[key=video_id]: dictionary of visual features (60, 512)
q_feat[key=question_id]: dictionary of features for questions

"""

data_path = 'your_dataset_path'

class AVQADataset(Dataset):

    def __init__(self, label):

        super().__init__()

        paths = [f'{data_path}/data/vggish_1fps', f'{data_path}/data/resnet_1fps']
        # paths = [f'{data_path}/data/vggish_1fps', f'{data_path}/data/r2plus1d_18']
        
        av_samples = os.listdir(paths[0])    # Exact same number of files for each dir
        a_feats = {}
        v_feats = {}
        sample_size = 1
        
        for av_sample in av_samples:
        # for av_sample in av_samples:

            a_feat = np.load(os.path.join(paths[0], av_sample))
            v_feat = np.load(os.path.join(paths[1], av_sample))
            # v_feat = np.load(f'{data_path}/data/visual_features/{av_sample}')

            video_id = av_sample[:-4]
            a_feats[video_id] = a_feat
            v_feats[video_id] = v_feat
            
        path = f'{data_path}/data/questions/avqa-{label}.json'
        q_samples = json.load(open(path, 'r'))

        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
        self.q_feats = []
        self.q_ids = []
        self.q_types = []
        self.q_anss = []
        self.vid_ids = []

        for q_sample in q_samples:
                    
            try:

                if a_feats[q_sample['video_id']].any():    # Process if corresponding audio and visual features exist
                    qs = q_sample['question_content']
                    if '<' in qs:
                        qs = replace_placeholders(qs, q_sample['templ_values'])
                    
                    self.q_feats.append(qs)
                    self.q_ids.append(q_sample['question_id'])
                    self.q_types.append(ast.literal_eval(q_sample['type']))
                    self.vid_ids.append(q_sample['video_id'])
                    self.q_anss.append(q_sample['anser'])

            except KeyError:
                pass

        self.a_feats = a_feats
        self.v_feats = v_feats
        
        self.ans_vocab = set(self.q_anss)
        self.vocab_dict = {id: ind for ind, id in enumerate(self.ans_vocab)}    # {'yes': 0, 'no': 1, ...}
    

    def __len__(self):

        return len(self.q_feats)

    def __getitem__(self, idx):
        
        video_id = self.vid_ids[idx]

        answer = self.q_anss[idx]
        
        ans_encoded = one_hot(answer, self.vocab_dict)

        question = self.q_feats[idx]
        
        encoded = self.tokenizer.encode_plus(question, padding='max_length', max_length=32, return_tensors='pt')
        input_ids = encoded['input_ids'][0]
        attn_mask = encoded['attention_mask'][0]
        
        sample = {

            'audio': self.a_feats[video_id],
            'visual': self.v_feats[video_id],
            'input_ids': input_ids,
            'attn_mask': attn_mask,
            'answer': ans_encoded,
            'video_id': video_id,
            'question_id': self.q_ids[idx],
            'question_type': self.q_types[idx]
        }

        return sample
