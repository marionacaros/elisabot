## Copyright 2019
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

import pandas as pd

CSV_data_directory = 'dataset/'


df_bing_train = pd.read_csv(CSV_data_directory + r'bing_train_all.csv', sep=',', header='infer')

df0 = pd.read_csv(CSV_data_directory + r'bing_train_all.csv', sep=',', header='infer')
df1 = pd.read_csv(CSV_data_directory + r'bing_test_all.csv', sep=',', header='infer')
df2 = pd.read_csv(CSV_data_directory + r'bing_val_all.csv', sep=',', header='infer')
df3 = pd.read_csv(CSV_data_directory + r'coco_test_all.csv', sep=',', header='infer')
df4 = pd.read_csv(CSV_data_directory + r'coco_train_all.csv', sep=',', header='infer')
df5 = pd.read_csv(CSV_data_directory + r'coco_val_all.csv', sep=',', header='infer')
df6 = pd.read_csv(CSV_data_directory + r'flickr_test_all.csv', sep=',', header='infer')
df7 = pd.read_csv(CSV_data_directory + r'flickr_train_all.csv', sep=',', header='infer')
df8 = pd.read_csv(CSV_data_directory + r'flickr_val_all.csv', sep=',', header='infer')

frames = [df0, df1, df2, df3, df4, df5, df6, df7, df8]

df_result = pd.concat(frames)

df_result.to_csv(CSV_data_directory + 'joined_df.csv', sep=';')
print('')