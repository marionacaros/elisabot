import pickle
import os
import json
import pandas as pd
from pandas import DataFrame
import requests
from pers_chat_processing.create_json import create_json
from tqdm import tqdm
import cv2
from skimage import io

from random import shuffle
from collections import OrderedDict

from create_input_files import create_input_files

CSV_data_directory = 'data/dataset/'
img_directory = '/projects/language/vqa2019/VQG/'
text_directory = 'data/dict/'
token_file = 'VQG.token.txt'


# ------------------------------------------------GET DATA FUNCTIONS --------------------------------------------------


def get_images_questions(df_img_questions):
    """ Download images and get questions
    returns list of images and questions"""

    print('Downloading data...')
    list_of_imqs = []
    not_available_img = []
    err_img_small = []

    for row in tqdm(df_img_questions.iterrows(), total=df_img_questions.shape[0]):

        # Debug
        # if i == 10:
        #    return list_of_imqs

        url = row[1]['image_url']
        # print("row 1:", row[1])
        try:
            response = requests.get(url)
            if response.status_code == 200:

                # ------------ Download images ------------
                # urllib.request.urlretrieve(url, img_directory + row[1]['image_id'] + '.jpg')
                # Try to open image
                img = io.imread(img_directory + row[1]['image_id'] + '.jpg')

                # Discard downloaded images not available
                if img.min() == 119 and img.max() == 255:
                    not_available_img.append(row[1]['image_id'])
                    print('NOT AVAILABLE IMAGE ', row[1]['image_id']+ '.jpg')
                    continue

                if img.size <= 8000:
                    err_img_small.append(row[1]['image_id'])
                    print('SMALL IMAGE ', row[1]['image_id']+ '.jpg')
                    continue

                # Try to resize the image
                img = cv2.imread(img_directory + row[1]['image_id'] + '.jpg')
                img = cv2.resize(img, (256, 256))

                # Debug
                # cv2.imwrite('sample_out_2.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                questions = row[1]['questions'].split("---")  # Getting multiple questions
                # print(questions)
                questions = [ques[:-1] for ques in questions]  # Removing "?" from every question to remove bias

                # Create list ['image_id', ['q1','q2','q3'] ],
                list_of_imqs.append([str(row[1]['image_id']), questions])

        except KeyboardInterrupt:
            print('Download cancelled.')
            break
        except:
            continue

    with open(os.path.join(text_directory, 'not_available_img' + '.json'), 'w') as j:
        json.dump(not_available_img, j)
    with open(os.path.join(text_directory, 'err_img_small' + '.json'), 'w') as j:
        json.dump(err_img_small, j)

    return list_of_imqs


# Shuffle
def shuffle_dict(dict):
    items = list(dict.items())
    shuffle(items)
    return OrderedDict(items)


# Generation of VQG.token.txt file
def vqg_token_generation(img_q_dic, img_capt_file='VQG.token.txt', train_s=0.8, test_s=0.1, val_s=0.1):
    text = ''
    i = 0
    num_img = int(len(img_q_dic))
    train_size = int(num_img * train_s)
    val_size = int(num_img * val_s)
    test_size = int(num_img * test_s)

    for img, questions in img_q_dic.items():

        if i < train_size:
            split = 'train'
        if train_size <= i < train_size + val_size:
            split = 'val'
        if i >= train_size + val_size:
            split = 'test'

        for indx, question in enumerate(questions):
            img_address = img + '.jpg'
            text += img_address + "#" + str(indx) + "\t" + str(i) + "\t" + split + "\t" + question + "\n"

        i += 1
    text = text[:-1]

    with open(text_directory + img_capt_file, 'w') as fh:
        fh.write(text)


# ---------------------------------------------------- MAIN ------------------------------------------------------

def main():
    # Read data from csv file
    data_df = DataFrame(pd.read_csv(CSV_data_directory + r'joined_df.csv', sep=';', header='infer'))

    df_img_questions = data_df.drop(columns=['query_term', 'captions'])
    # print(df_img_questions.head(5))

    # Download images and get questions
    list_of_imqs = get_images_questions(df_img_questions)

    # Generation of img_question_dict
    # key     =   val" \
    # "img_id  =   ['q1','q2','q3']"
    img_question_dict = dict()
    print('--- Generating image and questions dictionary ---')
    for img_id, qs in list_of_imqs:
        img_question_dict[str(img_id)] = qs

    # Shuffle dict
    print('--- Shuffling image and questions dictionary ---')
    img_question_dict_shuffle = shuffle_dict(img_question_dict)

    # if not os.path.exists(text_directory):
    #    os.mkdir(text_directory)
    #    print("Creating directory for storing info on img, questions and train,test splits")

    # Saving dictionary
    with open(text_directory + "img2questions_dict.pickle", "wb") as fh:
        pickle.dump(img_question_dict_shuffle, fh)

    with open(text_directory + "img2questions_dict.pickle", "rb") as fh:
        img_question_dict_shuffle = pickle.load(fh)

    # Generation of VQG.token.txt file
    vqg_token_generation(img_question_dict_shuffle, token_file)

    print('--- Token file generated ---')

    print('--- Creating json ---')
    # Write a json file containing all the captions of the images
    create_json(text_directory + token_file)

    print('--- json file generated ---')


if __name__ == '__main__':
    main()
