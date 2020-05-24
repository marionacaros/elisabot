
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
from torch import optim
import csv
import os
import codecs
from io import open
from chatbot_model_config import config
from train_chatbot import trimRareWords, EncoderRNN, LuongAttnDecoderRNN, trainIters
from utils_chatbot import loadPrepareData

directory = './'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "cornell-movie-dialogs"
corpus = os.path.join("data", "cornell-movie-dialogs")

model_name = 'cornell_model'
attn_model = 'dot'

# Checkpoints config
n_iteration = 8000
print_every = 50
save_every = 1000


# Configuration of models
hidden_size, encoder_n_layers, decoder_n_layers, dropout, batch_size, learning_rate, decoder_learning_ratio, \
teacher_forcing_ratio, clip, MAX_LENGTH,  MIN_COUNT = config()


def main():

    datafile = os.path.join('data/persona-chat', "data_pairs_v2.txt")

    # Load/Assemble voc and pairs
    save_dir = 'checkpoints_dialog_model'
    voc, _ = loadPrepareData("persona-chat-dialogs", datafile, save_dir)

    # cornell dialogs processing
    # data_processing()

    # Fine-tune with Cornell dataset
    datafile = os.path.join('data/cornell-movie-dialogs', "formatted_movie_lines.txt")
    voc_2, pairs_2 = loadPrepareData("cornell-movie-dialogs", datafile, save_dir)
    pairs_2 = trimRareWords(voc, pairs_2, MIN_COUNT)
    train_samples = int(len(pairs_2) * 0.8)
    pairs_train = pairs_2[0:train_samples]
    pairs_val = pairs_2[train_samples:]

    # Load model
    loadFilename = os.path.join('checkpoints_dialog_model/pers_model/persona-chat-dialogs/v7_1wf-12-65_500','{}_checkpoint.tar'.format(6000))

    # Load model if a loadFilename is provided
    if loadFilename:
        # If loading on same machine the model_VQG was trained on
        checkpoint = torch.load(loadFilename)
        # If loading a model_VQG trained on GPU to CPU
        # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']
        voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)
    if loadFilename:
        embedding.load_state_dict(embedding_sd)
    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)
    if loadFilename:
        encoder.load_state_dict(encoder_sd)
        decoder.load_state_dict(decoder_sd)
    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Ensure dropout layers are in train mode
    encoder.train()
    decoder.train()

    # Initialize optimizers
    print('Building optimizers ...')
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    if loadFilename:
        encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    # Run training iterations
    print("Start Training!")
    trainIters(model_name, voc, pairs_train, pairs_val, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)


def data_processing():
    # Define path to new file
    datafile = os.path.join(corpus, "formatted_movie_lines.txt")

    delimiter = '\t'
    # Unescape the delimiter
    delimiter = str(codecs.decode(delimiter, "unicode_escape"))

    # Initialize lines dict, conversations list, and field ids
    lines = {}
    conversations = []
    MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
    MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]

    # Load lines and process conversations
    print("\nProcessing corpus...")
    lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
    print("\nLoading conversations...")
    conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                      lines, MOVIE_CONVERSATIONS_FIELDS)

    # Write new csv file
    print("\nWriting newly formatted file...")
    with open(datafile, 'w', encoding='utf-8') as outputfile:
        writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
        for pair in extractSentencePairs(conversations):
            writer.writerow(pair)


# Splits each line of the file into a dictionary of fields
def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
            lines[lineObj['lineID']] = lineObj
    return lines


# Groups fields of lines from `loadLines` into conversations based on *movie_conversations.txt*
def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            # Extract fields
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
            # Convert string to list (convObj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            lineIds = eval(convObj["utteranceIDs"])
            # Reassemble lines
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
            conversations.append(convObj)
    return conversations


# Extracts pairs of sentences from conversations
def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        # Iterate over all the lines of the conversation
        for i in range(len(conversation["lines"]) - 1):  # We ignore the last line (no answer for it)
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            # Filter wrong samples (if one of the lists is empty)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, targetLine])
    return qa_pairs


if __name__ == '__main__':
    main()