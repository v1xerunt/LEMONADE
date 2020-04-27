import pickle
import numpy as np
import copy
import random
from sklearn.model_selection import train_test_split

random.seed(12345)

train_ehr = pickle.load(open('train_ehr','rb'))
train_label = pickle.load(open('train_label','rb'))
test_ehr = pickle.load(open('test_ehr','rb'))
test_label = pickle.load(open('test_label','rb'))

train_code = set()
for idx, patient in enumerate(train_ehr):
  for idx2, visit in enumerate(patient):
    for idx3, code in enumerate(visit):
      cur_code = code[:3]
      train_code.add(cur_code)
      
test_code = set()
for idx, patient in enumerate(test_ehr):
  for idx2, visit in enumerate(patient):
    for idx3, code in enumerate(visit):
      cur_code = code[:3]
      if cur_code not in train_code:
        test_code.add(cur_code)
        
all_code = train_code.union(test_code) #1739+84

disease_cnt = {}
for idx, patient in enumerate(train_ehr):
  if train_label[idx] == 1:
    for idx2, visit in enumerate(patient):
      for idx3, code in enumerate(visit):
        if code[:3] not in disease_cnt:
          disease_cnt[code[:3]] = {idx}
        else:
          disease_cnt[code[:3]].add(idx)
          
disease_cnt = {k: v for k, v in sorted(disease_cnt.items(), key=lambda item: len(item[1]), reverse=True)}
disease_list = list(disease_cnt.keys())

all_cnt = {}
for idx, patient in enumerate(train_ehr):
  for idx2, visit in enumerate(patient):
    for idx3, code in enumerate(visit):
      if code[:3] not in all_cnt:
        all_cnt[code[:3]] = {idx}
      else:
        all_cnt[code[:3]].add(idx)
for idx, patient in enumerate(test_ehr):
  for idx2, visit in enumerate(patient):
    for idx3, code in enumerate(visit):
      if code[:3] not in all_cnt:
        all_cnt[code[:3]] = {idx}
      else:
        all_cnt[code[:3]].add(idx)

for i in range(30):
  tmp_idx = set()
  tmp_idx2 = set()
  for j in range(i+1):
    tmp_idx = tmp_idx.union(all_cnt[disease_list[j]])
    tmp_idx2 = tmp_idx2.union(disease_cnt[disease_list[j]])
  tmp_rate = len(tmp_idx) / len(train_ehr)
  tmp_rate2 = len(tmp_idx2) / sum(train_label)
  print('Top %d: %.2f, %.2f'%(i+1, tmp_rate, tmp_rate2))
   
total_disease = list(all_cnt.keys())
random.shuffle(total_disease)

disease_idx = {}
for idx, code in enumerate(total_disease):
  disease_idx[code] = idx
    
train_dataset = []
train_type = []
train_scoring = []
for idx, patient in enumerate(train_ehr):
  done_list = list()
  for idx2, visit in reversed(list(enumerate(patient))):
    cur_type = []
    for idx3, code in enumerate(visit):
      for i in range(10):
        if code[:3] == disease_list[i] and idx2 != 0 and i not in done_list:
          done_list.append(i)
          cur_type.append(i)
    if len(cur_type) > 0:
      train_dataset.append(copy.deepcopy(patient[:idx2]))
      train_type.append(cur_type)
      train_scoring.append(copy.deepcopy(train_label[idx]))
          
for idx, patient in enumerate(train_dataset):
  for idx2, visit in enumerate(patient):
    tmp = np.zeros(1823, dtype=np.bool_)
    for idx3, code in enumerate(visit):
      tmp[disease_idx[code[:3]]] = 1
    patient[idx2] = tmp
    
pickle.dump(train_dataset[:35000],open('./dataset/train_dataset1','wb'))
pickle.dump(train_dataset[35000:70000],open('./dataset/train_dataset2','wb'))
pickle.dump(train_dataset[70000:],open('./dataset/train_dataset3','wb'))
pickle.dump(train_type, open('./dataset/train_type','wb'))
pickle.dump(train_scoring, open('./dataset/train_scoring','wb'))
pickle.dump(disease_idx, open('./dataset/disease_idx','wb'))

test_ehr, _, test_label, _ = train_test_split(test_ehr, test_label,test_size=0.5, random_state=12345, stratify=test_label)
test_ehr, valid_ehr, test_label, valid_label = train_test_split(test_ehr, test_label,test_size=0.2, random_state=12345, stratify=test_label)

test_dataset = []
test_type = []
test_scoring = []
for idx, patient in enumerate(test_ehr):
  done_list = list()
  for idx2, visit in reversed(list(enumerate(patient))):
    cur_type = []
    for idx3, code in enumerate(visit):
      for i in range(10):
        if code[:3] == disease_list[i] and idx2 != 0 and i not in done_list:
          done_list.append(i)
          cur_type.append(i)
    if len(cur_type) > 0:
      test_dataset.append(copy.deepcopy(patient[:idx2]))
      test_type.append(cur_type)
      test_scoring.append(copy.deepcopy(test_label[idx]))
          
for idx, patient in enumerate(test_dataset):
  for idx2, visit in enumerate(patient):
    tmp = np.zeros(1823, dtype=np.bool_)
    for idx3, code in enumerate(visit):
      tmp[disease_idx[code[:3]]] = 1
    patient[idx2] = tmp
    
pickle.dump(test_dataset[:40000],open('./dataset/test_dataset1','wb'))
pickle.dump(test_dataset[40000:],open('./dataset/test_dataset2','wb'))
pickle.dump(test_type, open('./dataset/test_type','wb'))
pickle.dump(test_scoring, open('./dataset/test_label','wb'))

valid_dataset = []
valid_type = []
valid_scoring = []
for idx, patient in enumerate(valid_ehr):
  done_list = list()
  for idx2, visit in reversed(list(enumerate(patient))):
    cur_type = []
    for idx3, code in enumerate(visit):
      for i in range(10):
        if code[:3] == disease_list[i] and idx2 != 0 and i not in done_list:
          done_list.append(i)
          cur_type.append(i)
    if len(cur_type) > 0:
      valid_dataset.append(copy.deepcopy(patient[:idx2]))
      valid_type.append(cur_type)
      valid_scoring.append(copy.deepcopy(test_label[idx]))
          
for idx, patient in enumerate(valid_dataset):
  for idx2, visit in enumerate(patient):
    tmp = np.zeros(1823, dtype=np.bool_)
    for idx3, code in enumerate(visit):
      tmp[disease_idx[code[:3]]] = 1
    patient[idx2] = tmp
    
pickle.dump(valid_dataset,open('./dataset/valid_dataset','wb'))
pickle.dump(valid_type, open('./dataset/valid_type','wb'))
pickle.dump(valid_scoring, open('./dataset/valid_label','wb'))