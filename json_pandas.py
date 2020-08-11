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

# with open('./datasets/vqa/v2_mscoco_train2014_annotations.json') as answer:
#     answer = json.load(answer)
#
# answer['annotations'][2]

"""
답을 이용하는거로 하면 train val 비교로해야 함
test셋은 답을 제공하지 않아서 test할 때 답을 이용하는 모델을 사용할 수 없음
"""

####
import cal_sim

with open('datasets/caption/train_cap.json') as train_cap:
    train_cap = json.load(train_cap)

with open('datasets/caption/val_cap.json') as val_cap:
    val_cap = json.load(val_cap)

with open('datasets/caption/test_cap.json') as test_cap:
    test_cap = json.load(test_cap)

df_train = pd.DataFrame(train_cap['data'])
df_val = pd.DataFrame(val_cap['data'])
df_test = pd.DataFrame(test_cap['data'])

df_train[:0]


df_train['similarity'] = cal_sim.sent_sim((df_train['question'], dtype=int32), (df_train['caption'], dtype=int32))

df_train.iloc[0]['question']


def txt2vec(sentence):
    s = sentence.split()
    tt = []
    for i in s:
        new_i = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            i.lower()
        )
        num = token_to_ix[new_i]
        tt.append(pretrained_emb[num])
    return tt



token_to_ix['what']
len(txt2vec(df_train.iloc[0]['question']))

df_train.iloc[0]['question']
df_train.iloc[0]['caption']

len(txt2vec(df_train.iloc[0]['caption']))

from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def word_sim(w1,w2): #word simiarity
    s = 0.5 * (1+ cos_sim(w1,w2))
    return s

def sent_sim(ss1, ss2): #sentence simiarity
    s1 = txt2vec(ss1)
    s2 = txt2vec(ss2)
    t = []
    for i in s1[2:]: #question   0,1 are PAD, UNK
        tmp = []
        for j in s2[2:]: #caption
            tmp_sim = word_sim(i,j)
            tmp.append(tmp_sim)
        t.append(max(tmp))
        sent_sim = sum(t) / len(s1[2:])
    return sent_sim


sent_sim(df_train.iloc[10]['question'], df_train.iloc[8]['caption'])

df_train.iloc[11]['question'] #유사도 좀 이상한 듯 너무 높게 나오는 것 같
df_train.iloc[8]['caption']