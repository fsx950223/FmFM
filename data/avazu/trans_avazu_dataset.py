# Copyright Verizon Media
# This project is licensed under the MIT. See license in project root for terms.

from tqdm import tqdm
from datetime import datetime
import json
import random

def trans_date_feat(datetime_str):
  date_hr = datetime.strptime(datetime_str, '%y%m%d%H')
  return date_hr.weekday(), date_hr.hour

def pre_parse_line(line, feat_list, fields_cnt = 23):
  splits = line.rstrip('\r\n').split(',', fields_cnt+2)

  datatime_str = splits[2]
  weekday, hour = trans_date_feat(datatime_str)
  features = [weekday, hour] + [int(val) for val in splits[3:5]] +\
             [int(val, 16) for val in splits[5:14]] + [int(val) for val in splits[14:]]

  for idx in range(0, fields_cnt):
    val = features[idx]
    if val not in feat_list[idx]:
      feat_list[idx][val] = 1
    else:
      feat_list[idx][val] += 1

  return

def parse_line(line, feat_list, fields_cnt):
  splits = line.rstrip('\r\n').split(',', fields_cnt+2)

  label = int(splits[1])
  vals = []

  datatime_str = splits[2]
  weekday, hour = trans_date_feat(datatime_str)
  features = [weekday, hour] + [int(val) for val in splits[3:5]] + \
             [int(val, 16) for val in splits[5:14]] + [int(val) for val in splits[14:]]

  for idx in range(0, fields_cnt):
    val = features[idx]
    if val not in feat_list[idx]:
      vals.append(0)
    else:
      vals.append(feat_list[idx][val])
  return label, vals

if __name__ == "__main__":
  thres = 5
  fields_cnt = 23

  data_file = '/scratch/datasets/avazu/train'
  train_out_file = open('train.csv', 'w')
  val_out_file = open('val.csv', 'w')
  test_out_file = open('test.csv', 'w')
  feature_index = open('feature_index', 'w')
  feature_json = open('features.json', 'w')

  dataset_ptr = open(data_file, 'r')
  titles = dataset_ptr.readline().rstrip('\r\n').split(',')
  titles[1] = 'weekday'
  del titles[0]

  dataset = dataset_ptr.readlines()

  feat_list = []
  for i in range(fields_cnt):
    feat_list.append({})

  for line in tqdm(dataset):
    pre_parse_line(line, feat_list, fields_cnt)

  for lst in tqdm(feat_list):
    idx = 1
    remove_keys = set()
    for key, val in lst.items():
      if val < thres:
        remove_keys.add(key)
      else:
        lst[key] = idx
        idx += 1
    for key in remove_keys:
      del lst[key]

  for idx, field in tqdm(enumerate(feat_list)):
    for feat, id in field.items():
      feature_index.write('%s\1|raw_feat_%s|\1%d\n' % (titles[idx], str(feat), id))
  feature_index.close()

  for line in tqdm(dataset):
    key, vals = parse_line(line, feat_list, fields_cnt)
    r = random.random()
    if r<0.7:
      train_out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))
    elif r>=0.7 and r<0.9:
      val_out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))
    else:
      test_out_file.write('%s,%s\n' % (key, ','.join([str(s) for s in vals])))

  feature_meta = []
  for idx in range(0, fields_cnt):
    feature_meta.append(('%s' % titles[idx], 'CATEGORICAL', 20))
  json.dump(feature_meta, feature_json, indent=2)

  train_out_file.close()
  val_out_file.close()
  test_out_file.close()
  del dataset

