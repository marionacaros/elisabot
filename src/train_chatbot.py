# -*- coding: utf-8 -*-

"""
Acknowledgments code use: Chatbot Tutorial **Author:** `Matthew Inkawhich <https://github.com/MatthewInkawhich>`_
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch as torch

from utils_chatbot import *
import datetime
from tensorboardX import SummaryWriter
import torch
import torch.utils.data
from torch import nn
from tqdm import tqdm
from chatbot_model_config import config
directory = './'

# Checkpoints config
n_iteration = 10000
print_every = 50
save_every = 1000

# Configuration of models
hidden_size, encoder_n_layers, decoder_n_layers, dropout, batch_size, learning_rate, decoder_learning_ratio, \
teacher_forcing_ratio, clip, MAX_LENGTH,  MIN_COUNT = config()

model_name = 'pers_model'
attn_model = 'dot'
# attn_model = 'general'
# attn_model = 'concat'

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None

# Cuda
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# Tensorboard location and plot names
now = datetime.datetime.now()
location = directory + 'runs/chatbot/' + now.strftime("%m-%d-%H:%M") + 'b' + str(batch_size) + 'lr' + \
           str(learning_rate)
writer = SummaryWriter(location)


def trimRareWords(voc, pairs, MIN_COUNT):
    # Trim words used under the MIN_COUNT from the voc
    voc.trim(MIN_COUNT)
    # Filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        # Check input sentence
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        # Check output sentence
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        # Only keep pairs that do not contain trimmed word(s) in their input or output sentence
        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total".format(len(pairs), len(keep_pairs),
                                                                len(keep_pairs) / len(pairs)))
    return keep_pairs


def maskNLLLoss(inp, target, mask):
    # Since we are dealing with batches of padded sequences, we cannot simply
    # consider all elements of the tensor when calculating loss. We define
    # ``maskNLLLoss`` to calculate our loss based on our decoder’s output
    # tensor, the target tensor, and a binary mask tensor describing the
    # padding of the target tensor. This loss function calculates the average
    # negative log likelihood of the elements that correspond to a *1* in the
    # mask tensor.

    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model_VQG weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def validate(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, batch_size):

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # No teacher forcing: next input is decoder's own current output
        _, topi = decoder_output.topk(1)
        decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
        decoder_input = decoder_input.to(device)
        # Calculate and accumulate loss
        mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss
        print_losses.append(mask_loss.item() * nTotal)
        n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    return sum(print_losses) / n_totals



def trainIters(model_name, voc, pairs_train, pairs_val, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding,
               encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip,
               corpus_name, loadFilename):
    # Load batches for each iteration
    training_batches = [batch2TrainData(voc, [random.choice(pairs_train) for _ in range(batch_size)])
                        for _ in range(n_iteration)]
    val_batches = [batch2TrainData(voc, [random.choice(pairs_val) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1

    if loadFilename:
        checkpoint = torch.load(loadFilename)
        start_iteration = checkpoint['iteration'] + 1

    print_loss = 0

    # Training loop
    print("Training...")
    for iteration in tqdm(range(start_iteration, n_iteration + 1)):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)

        print_loss += loss

        # Compute val loss every 100 iterations
        if iteration % 100 == 0 or iteration == 1:
            val_batch = val_batches[int(iteration) - 1]
            # Extract fields from batch
            input_variable_v, lengths_v, target_variable_v, mask_v, max_target_len_v = val_batch
            loss_val = validate(input_variable_v, lengths_v, target_variable_v, mask_v, max_target_len_v, encoder, decoder, batch_size)

            writer.add_scalar('loss_val', loss_val, iteration)

        # Save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name,
                                     '{}-{}-{}_{}'.format('v5_cornell', MAX_LENGTH, batch_size, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))

        if iteration == 1:
            # Tensorboard Logging - loss per iteration
            writer.add_scalar('loss_train', print_loss, iteration)

        if iteration % print_every == 0:
            # Tensorboard Logging - loss per iteration
            writer.add_scalar('loss_train', print_loss/print_every, iteration)
            print_loss = 0


def main():

    corpus_name = "persona-chat-dialogs"
    corpus = os.path.join("data", corpus_name)

    # Default word tokens
    PAD_token = 0  # Used for padding short sentences
    SOS_token = 1  # Start-of-sentence token
    EOS_token = 2  # End-of-sentence token

    # Define path to new file
    datafile = os.path.join('data/persona-chat', "data_pairs_v2.txt")

    # Load/Assemble voc and pairs
    save_dir = 'checkpoints_dialog_model'
    voc, pairs = loadPrepareData(corpus_name, datafile, save_dir)

    # Trim voc and pairs
    pairs = trimRareWords(voc, pairs, MIN_COUNT)
    train_samples = int(len(pairs) * 0.8)
    pairs_train = pairs[0:train_samples]
    pairs_val = pairs[train_samples:]

    print('---------------------------------------------------------------')
    print('LEN TRAIN PAIRS: ', len(pairs_train))
    print('Number iterations in 1 epoch: ', len(pairs_train) / batch_size)

    print('LEN VAL PAIRS: ', len(pairs_val))
    print('Number iterations in 1 epoch: ', len(pairs_val) / batch_size)

    ######################################################################
    # Run Model

    # loadFilename = os.path.join('checkpoints_dialog_model/pers_model/persona-chat-dialogs/v4-12-64_500','{}_checkpoint.tar'.format(2000))

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

    ######################################################################
    # Run Training
    # ~~~~~~~~~~~~

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
    print("Starting Training!")
    trainIters(model_name, voc, pairs_train, pairs_val, encoder, decoder, encoder_optimizer, decoder_optimizer,
               embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
               print_every, save_every, clip, corpus_name, loadFilename)


if __name__ == '__main__':
    main()