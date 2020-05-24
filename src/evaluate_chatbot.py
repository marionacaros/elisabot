
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from utils_chatbot import GreedySearchDecoder, normalizeString, indexesFromSentence, loadPrepareData, EncoderRNN, LuongAttnDecoderRNN
import torch
import os
import torch.nn as nn
from chatbot_model_config import config


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Configuration of models
hidden_size, encoder_n_layers, decoder_n_layers, dropout, _, _, _, _, _, MAX_LENGTH,  MIN_COUNT = config()

model_name = 'pers_model'
attn_model = 'dot'

checkpoint_iter = 14000

corpus_name = "persona-chat-dialogs"
save_dir = 'checkpoints_dialog_model'


def main():

    # Data path
    datafile = os.path.join('data/persona-chat', "data_pairs_v2.txt")

    print('Generate vocaulary...')
    voc, _ = loadPrepareData(corpus_name, datafile, save_dir)

    print('Loading model...')
    # Load model
    loadFilename = os.path.join('checkpoints_dialog_model/cornell_model/cornell-movie-dialogs/fine_tunned_cornell','{}_checkpoint.tar'.format(checkpoint_iter))
    # loadFilename = os.path.join('/work-nfs/mcaros/TFM/checkpoints_dialog_model/cornell_model/cornell-movie-dialogs'
    #                             '/v6_cornell-12-64_500-low_lr', '{}_checkpoint.tar'.format(checkpoint_iter))

    # loadFilename = os.path.join('checkpoints_dialog_model', 'pers_model', "persona-chat-dialogs",
    #                             '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
    #                             '{}_checkpoint.tar'.format(checkpoint_iter))

    # Load model if a loadFilename is provided
    checkpoint = torch.load(loadFilename)

    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')

    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)  # voc.num_words ,
    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')
    print(loadFilename)

    # Run Evaluation
    # ~~~~~~~~~~~~~~
    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)
    # evaluateInput(encoder, decoder, searcher, voc)

    # Open file with questions
    with open('/work-nfs/mcaros/TFM/questions_input_chatbot.txt') as f:
        lines = f.readlines()

    # Evaluate list of questions
    evaluateQuestions(encoder, decoder, searcher, voc, q_list=lines)


class TempModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 5, (3, 3))

    def forward(self, inp):
        return self.conv1(inp)

######################################################################
# Evaluate my text
# ~~~~~~~~~~~~~~~~
#
# Now that we have our decoding method defined, we can write functions for
# evaluating a string input sentence. The ``evaluate`` function manages
# the low-level process of handling the input sentence. We first format
# the sentence as an input batch of word indexes with *batch_size==1*. We
# do this by converting the words of the sentence to their corresponding
# indexes, and transposing the dimensions to prepare the tensor for our
# models. We also create a ``lengths`` tensor which contains the length of
# our input sentence. In this case, ``lengths`` is scalar because we are
# only evaluating one sentence at a time (batch_size==1). Next, we obtain
# the decoded response sentence tensor using our ``GreedySearchDecoder``
# object (``searcher``). Finally, we convert the response’s indexes to
# words and return the list of decoded words.
#
# ``evaluateInput`` acts as the user interface for our chatbot. When
# called, an input text field will spawn in which we can enter our query
# sentence. After typing our input sentence and pressing *Enter*, our text
# is normalized in the same way as our training data, and is ultimately
# fed to the ``evaluate`` function to obtain a decoded output sentence. We
# loop this process, so we can keep chatting with our bot until we enter
# either “q” or “quit”.
#
# Finally, if a sentence is entered that contains a word that is not in
# the vocabulary, we handle this gracefully by printing an error message
# and prompting the user to enter another sentence.


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):

    input_sentence = ''
    output_sentence = []

    while (1):
        # try:
        # Get input sentence
        input_sentence = input('> ')
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        # Normalize sentence
        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

        for word in output_words:
            if word == '.' or word == '!':
                output_sentence.append(word)
                break
            output_sentence.append(word)

        print('Bot:', ' '.join(output_sentence))
        output_sentence = []
        # except KeyError:
        #     print("Error: Encountered unknown word.")


def evaluateQuestions(encoder, decoder, searcher, voc, q_list):

    input_sentence = ''
    output_sentence = []

    for question in q_list:
        # Get input sentence
        input_sentence = question.rstrip()
        print(input_sentence)
        if input_sentence == '':
            continue
        # Check if it is quit case
        if input_sentence == 'q' or input_sentence == 'quit': break
        # Normalize sentence
        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]

        for word in output_words:
            if word == '.':
                break
            output_sentence.append(word)

        print('Bot:', ' '.join(output_sentence))
        output_sentence = []


def give_feedback(encoder, decoder, searcher, voc, input_sentence):

    try:
        # Normalize sentence
        input_sentence = normalizeString(input_sentence)
        # Evaluate sentence
        output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
        # Format and print response sentence
        output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
        print('Bot:', ' '.join(output_words))

    except KeyError:
        print("Error: Encountered unknown word.")

    return output_words


def load_chatbot_model(loadFilename):

    # Data path
    datafile = os.path.join('data/persona-chat', "data_pairs_v2.txt")

    print('Generate vocaulary...')
    voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)

    # Load model if a loadFilename is provided
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    # checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

    print('Building encoder and decoder ...')
    # Initialize word embeddings
    embedding = nn.Embedding(voc.num_words, hidden_size)  # voc.num_words ,

    embedding.load_state_dict(embedding_sd)

    # Initialize encoder & decoder models
    encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
    decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout)

    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

    # Use appropriate device
    encoder = encoder.to(device)
    decoder = decoder.to(device)
    print('Models built and ready to go!')

    # Set dropout layers to eval mode
    encoder.eval()
    decoder.eval()

    # Initialize search module
    searcher = GreedySearchDecoder(encoder, decoder)

    return encoder, decoder, searcher, voc


if __name__ == '__main__':
    main()
