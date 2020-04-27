import pandas as pd
import pickle
import numpy as np

df = pd.read_csv('./RareDiseaseDetection/data/ipf_train_data_100.csv')
df2 = pd.read_csv('./RareDiseaseDetection/data/ipf_test_data.csv')
convert_dict = pickle.load(open('convert_dict','rb'))

err_code = []

def gen_ehr(df):
  filtered_ehr = []
  filtered_label = []

  for idx, row in df.iterrows():
    cur_ehr = row['visits']
    cur_ehr = cur_ehr[2:-2]
    cur_ehr = cur_ehr.split('], [')
    new_ehr = []
    for idx, val in enumerate(cur_ehr):
      cur_ehr[idx] = cur_ehr[idx].split(', ')
      new_visit = []
      if 'DIAG_ICD9' in val or 'DIAG_ICD10' in val:
        for i in range(1, len(cur_ehr[idx]), 2):
          if cur_ehr[idx][i] == 'DIAG_ICD9':
            if cur_ehr[idx][i-1] not in convert_dict:
              if cur_ehr[idx][i-1]+'0' in convert_dict:
                new_visit.append(convert_dict[cur_ehr[idx][i-1]+'0'])
              elif cur_ehr[idx][i-1]+'1' in convert_dict:
                new_visit.append(convert_dict[cur_ehr[idx][i-1]+'1'])
              elif cur_ehr[idx][i-1]+'2' in convert_dict:
                new_visit.append(convert_dict[cur_ehr[idx][i-1]+'2'])
              elif cur_ehr[idx][i-1]+'00' in convert_dict:
                new_visit.append(convert_dict[cur_ehr[idx][i-1]+'00'])
              else:
                err_code.append(cur_ehr[idx][i-1])
            else:
              new_visit.append(convert_dict[cur_ehr[idx][i-1]])
          if cur_ehr[idx][i] == 'DIAG_ICD10':
            new_visit.append(cur_ehr[idx][i-1])
      if len(new_visit) > 0:
        new_ehr.append(new_visit)
    if len(new_ehr) > 0:
      filtered_ehr.append(new_ehr)
      filtered_label.append(0 if row['cohort'] == 'scoring' else 1)
  return filtered_ehr, filtered_label
  
train_ehr, train_label = gen_ehr(df)
pickle.dump(train_ehr, open('train_ehr','wb'))
pickle.dump(train_label, open('train_label','wb'))
test_ehr, test_label = gen_ehr(df2)
pickle.dump(test_ehr, open('test_ehr','wb'))
pickle.dump(test_label, open('test_label','wb'))