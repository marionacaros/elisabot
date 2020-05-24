import json
from tqdm import tqdm
import csv
import codecs


def main():
    with open('data/persona-chat/dic_train.json', 'r') as j:
        data_train = json.load(j)
    with open('data/persona-chat/dic_val.json', 'r') as k:
        data_val = json.load(k)

    print('Length data train: ', len(data_train))
    print('Length data val: ', len(data_val))

    datafile = 'data/persona-chat/formatted_data_val_pairs.txt'
    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extract_sentence_pairs(data_train):
            writer.writerow(pair)
        for pair in extract_sentence_pairs(data_val):
            writer.writerow(pair)

    # Remove some sentences


def extract_sentence_pairs(data_list):

    qa_pairs = []
    for dic in tqdm(data_list):
        for key in dic:
            if key == 'dialog':
                # Iterate over all the lines of the conversation
                for i in range(len(dic['dialog'])-1):
                    inputLine = dic['dialog'][i]
                    targetLine = dic['dialog'][i+1]

                    # Filter wrong samples (if one of the lists is empty)
                    if inputLine and targetLine:
                        qa_pairs.append([inputLine, targetLine])
    return qa_pairs


if __name__ == '__main__':
    main()


