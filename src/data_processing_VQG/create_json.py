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

import json

CSV_data_directory = '../dataset/'
img_directory = '//projects/language/vqa2019/VQG/'
directory = './'
text_directory = '../data/dict/'

token_file = text_directory + 'VQG.token.txt'


# Creating a dictionary containing all the captions of the images
def create_data_dict(token_file):

    captions = open(token_file, 'r').read().strip().split('\n')
    images = {'filename': ''}
    img_v = []
    sentences = []
    imgid = '0'

    for i, row in enumerate(captions):
        row = row.split('\t')
        filename = row[0][:len(row[0]) - 2]

        # Save sentences of previous image
        if imgid != row[1]:
            images['sentences'] = sentences
            img_v.append(images)
            images = {'filename': ''}
            sentences = []

        # Get images info
        imgid = row[1]
        num_s = int(row[0][-1])
        split = row[2]
        tokens = (row[3]).split()

        sentences.append({'tokens': tokens, 'raw': row[3], 'imgid': imgid})

        if filename not in images['filename']:
            images['filepath'] = img_directory
            images['filename'] = filename
            images['imgid'] = imgid
            images['split'] = split

    return img_v


# ----- Writing JSON file ------
def write_json(data, path=text_directory, file_name='dataset'):
    file = './' + path + '/' + file_name + '.json'
    with open(file, 'w') as fp:
        json.dump(data, fp)


def create_json(token_file):
    # Creating a dictionary containing all the captions of the images
    images_vector = create_data_dict(token_file)
    d = {'images': images_vector}

    print('--- Writing JSON file ---')
    write_json(d)


def main():
    create_json(token_file)


if __name__ == '__main__':
    main()
