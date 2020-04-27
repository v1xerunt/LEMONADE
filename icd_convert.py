import csv
import pickle

convert_dict = {}

with open('icd_convert.csv',newline='') as f:
  reader = csv.reader(f)
  next(f)
  for row in reader:
    convert_dict[row[0]] = row[1]
    
pickle.dump(convert_dict, open('convert_dict','wb'))