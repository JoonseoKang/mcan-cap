"""
json 불러와서 캡션 붙이는 것
"""

import json
import pandas as pd

path = './datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json'

with open(path) as question:
    question = json.load(question)

question['questions'][0]
question['questions'][1]
question['questions'][2]

df = pd.DataFrame(question['questions'])
df

caption_path = './datasets/caption/vis_st_trainval.json'

with open(caption_path) as cap:
    cap = json.load(cap)

df_cap = pd.DataFrame(cap)
df_cap

df_addcap = pd.merge(df, df_cap, how='left', on='image_id')
df_addcap[:0]
del df_addcap['file_path']

########################################################################################################################
"""
pandas to json
"""
df_addcap.to_json('./datasets/caption/train_cap.json', orient='table')

with open('./datasets/caption/train_cap.json') as train_cap:
    train_cap = json.load(train_cap)
########################################################################################################################
"""val test도 마찬가지"""로

path = './datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json'

with open(path) as question:
    question = json.load(question)

df = pd.DataFrame(question['questions'])
df

caption_path = './datasets/caption/vis_st_trainval.json'

with open(caption_path) as cap:
    cap = json.load(cap)

df_cap = pd.DataFrame(cap)
df_cap

df_addcap = pd.merge(df, df_cap, how='left', on='image_id')
df_addcap[:0]
del df_addcap['file_path']

df_addcap.to_json('./datasets/caption/val_cap.json', orient='table')

#test
path = './datasets/vqa/v2_OpenEnded_mscoco_test2015_questions.json'

with open(path) as question:
    question = json.load(question)

df = pd.DataFrame(question['questions'])
df
df['image_id'] = df.image_id.astype(int)
caption_path = './datasets/caption/vis_st_test.json'

with open(caption_path) as cap:
    cap = json.load(cap)

df_cap = pd.DataFrame(cap)
df_cap
df_cap['image_id'] = df_cap.image_id.astype(int)

df_addcap = pd.merge(df, df_cap, how='left', on='image_id')
df_addcap[:0]

del df_addcap['file_path']

df_addcap.to_json('./datasets/caption/test_cap.json', orient='table')
########################################################################################################################
from core.data.ans_punct import prep_ans
import numpy as np
import en_vectors_web_lg, random, re, json
import json
from core.data.data_utils import ques_load

stat_ques_list = \
            json.load(open('./datasets/caption/train_cap.json', 'r'))['data'] + \
            json.load(open('./datasets/caption/val_cap.json', 'r'))['data'] + \
            json.load(open('./datasets/caption/test_cap.json', 'r'))['data']

def tokenize(stat_ques_list, use_glove):
    token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }

    spacy_tool = None
    pretrained_emb = []
    if use_glove:
        spacy_tool = en_vectors_web_lg.load()
        pretrained_emb.append(spacy_tool('PAD').vector)
        pretrained_emb.append(spacy_tool('UNK').vector)

    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)
    for ques in stat_ques_list:
        words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            ques['caption'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

        for word in words:
            if word not in token_to_ix:
                token_to_ix[word] = len(token_to_ix)
                if use_glove:
                    pretrained_emb.append(spacy_tool(word).vector)

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

token_to_ix, pretrained_emb = tokenize(stat_ques_list, True)
#######################################################################################################################