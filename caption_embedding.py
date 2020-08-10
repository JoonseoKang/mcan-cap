from core.data.ans_punct import prep_ans
import numpy as np
import en_vectors_web_lg, random, re, json
import json
from core.data.data_utils import ques_load

img_feat_path_list = []
# split_list = __C.SPLIT[__C.RUN_MODE].split('+')

stat_ques_list = \
            json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions'] + \
            json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions'] + \
            json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))['questions']


# qid_to_ques = ques_load(ques_list)

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

    pretrained_emb = np.array(pretrained_emb)

    return token_to_ix, pretrained_emb

token_to_ix, pretrained_emb = tokenize(stat_ques_list, True)

token_to_ix
len(pretrained_emb)

stat_ques_list[1]
words = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            stat_ques_list[2]['question'].lower()
        ).replace('-', ' ').replace('/', ' ').split()

token_to_ix = {
        'PAD': 0,
        'UNK': 1,
    }
spacy_tool = None
pretrained_emb = []
spacy_tool = en_vectors_web_lg.load()
pretrained_emb.append(spacy_tool('PAD').vector)
pretrained_emb.append(spacy_tool('UNK').vector)

for word in words:
    if word not in token_to_ix:
        token_to_ix[word] = len(token_to_ix)
        pretrained_emb.append(spacy_tool(word).vector)

pretrained_emb = np.array(pretrained_emb)

tmp_cap ="a baseball player standing next to home plate"

words_cap = re.sub(
            r"([.,'!?\"()*#:;])",
            '',
            tmp_cap.lower()
        ).replace('-', ' ').replace('/', ' ').split()

token_to_ix_cap = {
        'PAD': 0,
        'UNK': 1,
    }

pretrained_emb_cap = []
pretrained_emb_cap.append(spacy_tool('PAD').vector)
pretrained_emb_cap.append(spacy_tool('UNK').vector)

for word in words_cap:
    if word not in token_to_ix_cap:
        token_to_ix_cap[word] = len(token_to_ix_cap)
        pretrained_emb_cap.append(spacy_tool(word).vector)

pretrained_emb_cap = np.array(pretrained_emb_cap)

pretrained_emb[2]
pretrained_emb_cap[2]

##################################유사도 구하기############################################################################

# len(words)
# len(words_cap)
# len(pretrained_emb)
# len(pretrained_emb[2:])
#
#
# tt = []
# for i in pretrained_emb[2:]:
#     tmp = []
#     for j in pretrained_emb_cap[2:]:
#         tmtm = word_sim(i,j)
#         tmp.append(tmtm)
#     print(max(tmp))
#     tt.append(max(tmp))
#     s = sum(tt)
#     sim = s / len(pretrained_emb[2:])
#
# sim

from numpy import dot
from numpy.linalg import norm
import numpy as np

def cos_sim(A, B):
    return dot(A, B) / (norm(A) * norm(B))

def word_sim(w1,w2): #word simiarity
    s = 0.5 * (1+ cos_sim(w1,w2))
    return s

def sent_sim(s1, s2): #sentence simiarity
    t = []
    for i in s1[2:]: #question   0,1 are PAD, UNK
        tmp = []
        for j in s2[2:]: #caption
            tmp_sim = word_sim(i,j)
            tmp.append(tmp_sim)
        t.append(max(tmp))
        sent_sim = sum(t) / len(s1[2:])
    return sent_sim




