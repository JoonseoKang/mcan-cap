import json
import pandas as pd

with open('datasets/caption/train_cap.json') as train_cap:
    train_cap = json.load(train_cap)

with open('datasets/caption/val_cap.json') as val_cap:
    val_cap = json.load(val_cap)

with open('datasets/caption/test_cap.json') as test_cap:
    test_cap = json.load(test_cap)



# df_tv = pd.concat([df_train, df_val], ignore_index=True)
# df_tv = df_tv.drop(['image_id', 'question_id'], axis='columns')
# df_tv = df_tv.sort_values(by='similarity',ascending=False)
#
# df_tv[df_tv.similarity > 0.8] #324275개

######
for i in train_cap['data']:
    # print(i)
    if i['similarity'] < 0.8:
        i['caption'] = ''

for i in val_cap['data']:
    # print(i)
    if i['similarity'] < 0.8:
        i['caption'] = ''

for i in test_cap['data']:
    # print(i)
    if i['similarity'] < 0.8:
        i['caption'] = ''



#############3
"""Question + Caption """

for i in train_cap['data']:
    i['question'] = i['question']+ ' ' + i['caption']

for i in val_cap['data']:
    i['question'] = i['question']+ ' ' + i['caption']

for i in test_cap['data']:
    i['question'] = i['question']+ ' ' + i['caption']



with open('datasets/caption/train_cap_under08.json', 'w') as f:
    json.dump(train_cap, f)

with open('datasets/caption/val_cap_under08.json', 'w') as f2:
    json.dump(val_cap, f2)

with open('datasets/caption/test_cap_under08.json', 'w') as f3:
    json.dump(test_cap, f3)

df_train = pd.DataFrame(train_cap['data'])
df_val = pd.DataFrame(val_cap['data'])
df_test = pd.DataFrame(test_cap['data'])

df_train.iloc[0]
del df_train['caption']
del df_train['similarity']

del df_val['caption']
del df_val['similarity']

del df_test['caption']
del df_test['similarity']

df_train.to_json('datasets/caption/train_under08.json', orient='table')

with open('datasets/caption/train_under08.json') as train_cap:
    train_cap = json.load(train_cap)

for i in train_cap['data']:
    del i['index']

df_val.to_json('datasets/caption/val_under08.json', orient='table')

with open('datasets/caption/val_under08.json') as val_cap:
    val_cap = json.load(val_cap)

for i in val_cap['data']:
    del i['index']

val_cap

df_test.to_json('datasets/caption/test_under08.json', orient='table')
with open('datasets/caption/test_under08.json') as test_cap:
    test_cap = json.load(test_cap)

for i in test_cap['data']:
    del i['index']

with open('datasets/caption/train_under08.json', 'w') as f:
    json.dump(train_cap, f)

with open('datasets/caption/val_under08.json', 'w') as f2:
    json.dump(val_cap, f2)

with open('datasets/caption/test_under08.json', 'w') as f3:
    json.dump(test_cap, f3)


"""
path_cfgs.py 경로 바꾸고
base_cfgs.py에서 MAX_TOKEN 숫자 바꾸고 
load_data.py에서 questions -> data로 바꾸고
"""