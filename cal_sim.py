from core.data.ans_punct import prep_ans
import numpy as np
import en_vectors_web_lg, random, re, json
import json
from core.data.data_utils import ques_load
import pandas as pd
from numpy import dot
from numpy.linalg import norm
import numpy as np

stat_ques_list = \
            json.load(open('./datasets/caption/train_cap2.json', 'r'))['data'] + \
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


with open('datasets/caption/train_cap2.json') as train_cap:
    train_cap = json.load(train_cap)

with open('datasets/caption/val_cap.json') as val_cap:
    val_cap = json.load(val_cap)

with open('datasets/caption/test_cap.json') as test_cap:
    test_cap = json.load(test_cap)

# df_train = pd.DataFrame(train_cap['data'])
# df_val = pd.DataFrame(val_cap['data'])
# df_test = pd.DataFrame(test_cap['data'])
from core.data.ans_punct import prep_ans


def txt2vec(sentence):
    # s = sentence.split()
    tt = []

    new_i = re.sub(
        r"([.,'!?\"()*#:;])",
        '',
        sentence.lower()
    ).replace('-', ' ').replace('/', ' ').split()

    for i in new_i:
        if i in token_to_ix:
            num = token_to_ix[i]
            tt.append(pretrained_emb[num])
        else:
            # num = token_to_ix['UNK']
            tt.append(pretrained_emb[1])
    return tt


# bundesbahn
# txt2vec('bundesbahn')
# token_to_ix['bundesbahn']

# txt2vec('junseo')

def cos_sim(A, B):
    return np.matmul(A, np.transpose(B)) / (norm(A) * norm(B))


def word_sim(w1,w2): #word simiarity
    s = 0.5 * (1+ cos_sim(w1,w2))
    return s

# word_sim(pretrained_emb[1281], pretrained_emb[2154])
# token_to_ix['bad']
# token_to_ix['good']
# cos_sim(pretrained_emb[1073], pretrained_emb[168])
# word_sim(pretrained_emb[1073], pretrained_emb[168])

def sent_sim(sent1, sent2):
    sent2vec1 = txt2vec(sent1) #question
    sent2vec2 = txt2vec(sent2) #caption
    global sent_similarity
    sent_tmp = []

    for i in sent2vec1:
        vec_tmp = []
        for j in sent2vec2:
            tmp_sim = word_sim(i, j)
            vec_tmp.append(tmp_sim)
        sent_tmp.append(max(vec_tmp))

        sent_similarity = sum(sent_tmp) / len(sent2vec1)

    return sent_similarity


# sent_sim('is there a apple?', 'an apple is on the table')
# sent_sim('hello', 'hello')
# train_cap['data'][0]['question']
# train_cap['data'][0]['caption']
# sent_sim(train_cap['data'][0]['question'], train_cap['data'][0]['caption'])
# train_cap['data'][0]['similarity'] = sent_sim(train_cap['data'][0]['question'], train_cap['data'][0]['caption'])

######train_answer similarity########
# with open('./datasets/caption/train_qacap.json') as train_qacap:
#     train_qacap = json.load(train_qacap)
#
# # def ans_stat(json_file):
# #     ans_to_ix, ix_to_ans = json.load(open(json_file, 'r'))
# #
# #     return ans_to_ix, ix_to_ans
#
# # ans_to_ix, ix_to_ans = ans_stat('./core/data/answer_dict.json')
#
# for i in train_qacap['data']:
#     i['answer'] = prep_ans(i['multiple_choice_answer'])
#
# for i in train_qacap['data']:
#     del i['multiple_choice_answer']
#     # del i['index']
# with open('datasets/caption/train_qacap.json', 'w') as f:
#     json.dump(train_qacap, f)

with open('./datasets/caption/val_qacap.json') as val_qacap:
    val_qacap = json.load(val_qacap)

for i in val_qacap['data']:
    i['q_similarity'] = sent_sim(i['question'], i['caption'])

for i in val_qacap['data']:
    i['a_similarity'] = sent_sim(i['multiple_choice_answer'], i['caption'])



for i in val_qacap['data']:
    i['total_similarity'] = (i['a_similarity'] + i['q_similarity'] ) / 2


with open('datasets/caption/val_qacap_sim.json', 'w') as f:
    json.dump(val_qacap, f)


with open('./datasets/caption/train_qacap_sim.json') as train_qacap:
    train_qacap = json.load(train_qacap)

df_sim = pd.DataFrame(val_qacap['data'])

# import matplotlib.pyplot as plt
# # plt.hist([df_sim['similarity'], df_val['similarity'], df_test['similarity']], label=['train', 'val', 'test'])
# plt.hist(df_sim['a_similarity'],color='blue', label='train', alpha=0.5)
# plt.hist(df_sim['q_similarity'],color='red', label='val', alpha=0.5)
# plt.hist(df_sim['total_similarity'], color='green', label='test', alpha=0.5)
# plt.legend(loc='upper right')
# plt.show()

df_sim2 = df_sim.sort_values(by='total_similarity',ascending=False)

df_sim2['total_similarity'].isnull().sum()

df_sim2.iloc[0]
del df_sim2['index']
del df_sim2['image_id']
del df_sim2['question']
del df_sim2['question_id']
del df_sim2['caption']
del df_sim2['multiple_choice_answer']


df_sim2.describe()


#####################################


for i in train_cap['data']:
    i['q_similarity'] = sent_sim(i['question'], i['caption'])

for i in val_cap['data']:
    i['similarity'] = sent_sim(i['question'], i['caption'])

for i in test_cap['data']:
    i['similarity'] = sent_sim(i['question'], i['caption'])

for i in train_cap['data']:
    del i['index']

for i in val_cap['data']:
    del i['index']

for i in test_cap['data']:
    del i['index']


with open('datasets/caption/train_cap.json', 'w') as f:
    json.dump(train_cap, f)

with open('datasets/caption/val_cap.json', 'w') as f2:
    json.dump(val_cap, f2)

with open('datasets/caption/test_cap.json', 'w') as f3:
    json.dump(test_cap, f3)

########################################################################################################################
"""similarity distribution check"""
df_train = pd.DataFrame(train_cap['data'])
df_val = pd.DataFrame(val_cap['data'])
df_test = pd.DataFrame(test_cap['data'])

import matplotlib.pyplot as plt
plt.hist([df_train['similarity'], df_val['similarity'], df_test['similarity']], label=['train', 'val', 'test'])
# plt.hist(df_train['similarity'],color='blue', label='train', alpha=0.5)
# plt.hist(df_val['similarity'],color='red', label='val', alpha=0.5)
# plt.hist(df_test['similarity'], color='green', label='test', alpha=0.5)
plt.legend(loc='upper right')
plt.show()
########################################################################################################################
"""train+val"""
df_tv = pd.concat([df_train, df_val], ignore_index=True)
df_tv = df_tv.drop(['image_id', 'question_id'], axis='columns')
df_tv = df_tv.sort_values(by='similarity',ascending=False)
df_tv['similarity'].isnull().sum() #413개
df_tv = df_tv.fillna(0)
df_tv.describe() # q3 is 0.83 mean is 0.80 q1 is 0.77



"""similarity check"""
# df_t = df_train.drop(['image_id', 'question_id'], axis='columns')
# df_t.sort_values(by='similarity')
# df_t['similarity'].isnull().sum() #275개
# df_t = df_t.fillna(0)

# df_t = df_t.sort_values(by='similarity', ascending=False)
# df_t.iloc[0]
# df_t.iloc[15]
#
# df_t.describe()

# sent_sim('Where are they riding a skylift?', 'a man and a woman posing for a picture')
#
# txt2vec('skylift') 모두 0 인 matrix
# word_sim(txt2vec('skylift'), txt2vec('picture'))
########################################################################################################################
# from core.data.ans_punct import prep_ans
# import numpy as np
# import en_vectors_web_lg, random, re, json
# import json
# from core.data.data_utils import ques_load

# img_feat_path_list = []
# # split_list = __C.SPLIT[__C.RUN_MODE].split('+')
#
# stat_ques_list = \
#             json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_train2014_questions.json', 'r'))['questions'] + \
#             json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_val2014_questions.json', 'r'))['questions'] + \
#             json.load(open('./datasets/vqa/v2_OpenEnded_mscoco_test2015_questions.json', 'r'))['questions']
#
#
# # qid_to_ques = ques_load(ques_list)
#
# def tokenize(stat_ques_list, use_glove):
#     token_to_ix = {
#         'PAD': 0,
#         'UNK': 1,
#     }
#
#     spacy_tool = None
#     pretrained_emb = []
#     if use_glove:
#         spacy_tool = en_vectors_web_lg.load()
#         pretrained_emb.append(spacy_tool('PAD').vector)
#         pretrained_emb.append(spacy_tool('UNK').vector)
#
#     for ques in stat_ques_list:
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ques['question'].lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for word in words:
#             if word not in token_to_ix:
#                 token_to_ix[word] = len(token_to_ix)
#                 if use_glove:
#                     pretrained_emb.append(spacy_tool(word).vector)
#
#     pretrained_emb = np.array(pretrained_emb)
#
#     return token_to_ix, pretrained_emb
#
# token_to_ix, pretrained_emb = tokenize(stat_ques_list, True)
#
# token_to_ix
# len(pretrained_emb)
#
# stat_ques_list[1]
# words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             stat_ques_list[2]['question'].lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
# token_to_ix = {
#         'PAD': 0,
#         'UNK': 1,
#     }
# spacy_tool = None
# pretrained_emb = []
# spacy_tool = en_vectors_web_lg.load()
# pretrained_emb.append(spacy_tool('PAD').vector)
# pretrained_emb.append(spacy_tool('UNK').vector)
#
# for word in words:
#     if word not in token_to_ix:
#         token_to_ix[word] = len(token_to_ix)
#         pretrained_emb.append(spacy_tool(word).vector)
#
# pretrained_emb = np.array(pretrained_emb)
#
# tmp_cap ="a baseball player standing next to home plate"
#
# words_cap = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             tmp_cap.lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
# token_to_ix_cap = {
#         'PAD': 0,
#         'UNK': 1,
#     }
#
# pretrained_emb_cap = []
# pretrained_emb_cap.append(spacy_tool('PAD').vector)
# pretrained_emb_cap.append(spacy_tool('UNK').vector)
#
# for word in words_cap:
#     if word not in token_to_ix_cap:
#         token_to_ix_cap[word] = len(token_to_ix_cap)
#         pretrained_emb_cap.append(spacy_tool(word).vector)
#
# pretrained_emb_cap = np.array(pretrained_emb_cap)
#
# pretrained_emb[2]
# pretrained_emb_cap[2]

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
##########################################################
# from core.data.ans_punct import prep_ans
# import numpy as np
# import en_vectors_web_lg, random, re, json
# import json
# from core.data.data_utils import ques_load
#
# stat_ques_list = \
#             json.load(open('./datasets/caption/train_cap.json', 'r'))['data'] + \
#             json.load(open('./datasets/caption/val_cap.json', 'r'))['data'] + \
#             json.load(open('./datasets/caption/test_cap.json', 'r'))['data']
#
# def tokenize(stat_ques_list, use_glove):
#     token_to_ix = {
#         'PAD': 0,
#         'UNK': 1,
#     }
#
#     spacy_tool = None
#     pretrained_emb = []
#     if use_glove:
#         spacy_tool = en_vectors_web_lg.load()
#         pretrained_emb.append(spacy_tool('PAD').vector)
#         pretrained_emb.append(spacy_tool('UNK').vector)
#
#     for ques in stat_ques_list:
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ques['question'].lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for word in words:
#             if word not in token_to_ix:
#                 token_to_ix[word] = len(token_to_ix)
#                 if use_glove:
#                     pretrained_emb.append(spacy_tool(word).vector)
#     for ques in stat_ques_list:
#         words = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             ques['caption'].lower()
#         ).replace('-', ' ').replace('/', ' ').split()
#
#         for word in words:
#             if word not in token_to_ix:
#                 token_to_ix[word] = len(token_to_ix)
#                 if use_glove:
#                     pretrained_emb.append(spacy_tool(word).vector)
#
#     pretrained_emb = np.array(pretrained_emb)
#
#     return token_to_ix, pretrained_emb
#
# token_to_ix, pretrained_emb = tokenize(stat_ques_list, True)
#
# ###########################################################
# from numpy import dot
# from numpy.linalg import norm
# import numpy as np
#
# def cos_sim(A, B):
#     return dot(A, B) / (norm(A) * norm(B))
#
# def word_sim(w1,w2): #word simiarity
#     s = 0.5 * (1+ cos_sim(w1,w2))
#     return s
#
#
# def txt2vec(sentence):
#     s = sentence.split()
#     tt = []
#     for i in s:
#         new_i = re.sub(
#             r"([.,'!?\"()*#:;])",
#             '',
#             i.lower()
#         )
#         num = token_to_ix[new_i]
#         tt.append(pretrained_emb[num])
#     return tt
#
# def sent_sim(ss1, ss2): #sentence simiarity
#     s1 = txt2vec(ss1)
#     s2 = txt2vec(ss2)
#     t = []
#     for i in s1[2:]: #question   0,1 are PAD, UNK
#         tmp = []
#         for j in s2[2:]: #caption
#             tmp_sim = word_sim(i,j)
#             tmp.append(tmp_sim)
#         t.append(max(tmp))
#         sent_sim = sum(t) / len(s1[2:])
#     return sent_sim
#
#
#

# sent = 'i like a girl'
# s = sent.split()
# s[0]
# token_to_ix[s[0]]
# pretrained_emb[token_to_ix[s[0]]]

# a = txt2vec('i like a girl')
# b = txt2vec('a girl is standing')
