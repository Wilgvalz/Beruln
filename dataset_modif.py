from datasets import load_dataset, DatasetDict
import os
import pandas as pd

dataset = load_dataset("json", data_files="diversevul_20230702.json")

# Split into train/valid/test
train_valid = dataset["train"].train_test_split(test_size=0.2, seed=0, train_indices_cache_file_name="train.indices")
train_data, valid_data = train_valid["train"], train_valid["test"]
valid_test = valid_data.train_test_split(test_size=0.5, seed=0, train_indices_cache_file_name="valid.indices", test_indices_cache_file_name="test.indices")
valid_data, test_data = valid_test["train"], valid_test["test"]
dataset = DatasetDict({
    "train": train_data,
    "validation": valid_data,
    "test": test_data,
})
#.....

from collections import defaultdict
dict1 = {}
dict1 = defaultdict(list)
print(1)
dict2 = {}
dict2 = defaultdict(list)
m = 0
n = 0
w = 0
b = 0
s = 0
for d in dataset['train']:
   
    if d['target'] == int(1) and w <= 9999:
        
        dict1['func'].append(d['func']) 
        dict1['target'].append(d['target'])
        w += 1
    elif d['target'] == int(0) and b <= 9999:
   
        dict2['func'].append(d['func']) 
        dict2['target'].append(d['target'])
        b += 1
    
#.....
dict11 = {}
dict11 = defaultdict(list)
print(1)
dict21 = {}
dict21 = defaultdict(list)
m = 0
n = 0
w = 0
b = 0
s = 0
for d in dataset['validation']:
   
    if d['target'] == int(1) and w <= 1799:
        
        dict11['func'].append(d['func']) 
        dict11['target'].append(d['target'])
        w += 1
    if d['target'] == int(0) and b <= 1799:
   
        dict21['func'].append(d['func']) 
        dict21['target'].append(d['target'])
        b += 1

#.....

def mergeDict(dict1, dict2):
    for k, v in dict2.items():
        if dict1.get(k):
            dict1[k] = [dict1[k], v]
        else:
            dict1[k] = v        
    return dict1

dict_train = mergeDict(dict1, dict2)
dict_val = mergeDict(dict11, dict21) 

#.....

dict_trr_target = dict_train['target'][0]
dict_trr_func = dict_train['func'][0]
dict_val_target = dict_val['target'][0]
dict_val_func = dict_val['func'][0]
#dict_tr = defaultdict(list)

print(type(dict_trr_target))
for i in range(10000):
       
        dict_trr_target.append(dict_train['target'][1][i])
        dict_trr_func.append(dict_train['func'][1][i])
        
for i in range(1800): 
        dict_val_target.append(dict_val['target'][1][i])
        dict_val_func.append(dict_val['func'][1][i])
dict_tr_target = dict_trr_target 
dict_tr_func = dict_trr_func
#.....
for i in range(3600):
        dict_tr_target.append(dict_val_target[i])
        dict_tr_func.append(dict_val_func[i])

df2 = pd.DataFrame({'func': dict_tr_func, 'target': dict_tr_target})
df2.to_csv("dataset_dv_my.csv")    
