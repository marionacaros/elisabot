"""
Acknowledgments code use:  <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning>`_
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import skimage.transform
import argparse
import string
from scipy.misc import imread, imresize
from PIL import Image
from model_VQG.models import Encoder, DecoderWithAttention
import random

import sys
sys.path.insert(0, './checkpoints')

IMAGE_FOLDER = '//projects/language/vqa2019/VQG/'
img_path = '110111.jpg'
test_data_file = 'data/my_test_filenames.txt'
MY_TEST_FOLDER = 'data/my_test_images/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():

    word_map = './data/caption_data/WORDMAP_VQG_5_cap_per_img_2_min_word_freq.json'
    model_enc =  'checkpoints/6w_2wf_VQG_model/13epoch/enc_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'
    model_dec = 'checkpoints/6w_2wf_VQG_model/13epoch/dec_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'

    # Load word map (word2ix)
    with open(word_map, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    beam_size = 7

    # Model
    encoder = Encoder()
    decoder = DecoderWithAttention(len(word_map))

    # Load model
    encoder.load_state_dict(torch.load(model_enc))
    decoder.load_state_dict(torch.load(model_dec))
    decoder = decoder.to(device)
    decoder.eval()
    encoder = encoder.to(device)
    encoder.eval()

    # img = get_random_pic()
    # print(img)

    # File with images names
    with open(test_data_file) as f:
        lines = f.readlines()

    for img_path in lines:
        img = MY_TEST_FOLDER + img_path.rstrip()

        # Encode, decode with attention and beam search
        alphas, ordered_seq, all_sequences = caption_image_beam_search(encoder, decoder, img, word_map, beam_size)

        # Decode index to word
        ordered_questions_token = []
        for question in ordered_seq:
            q = [rev_word_map[ind] for ind in question[1:-1]]
            # Remove <unk>
            if '<unk>' in q:
                pass
            else:
                ordered_questions_token.append(q)

        # Detokenize question
        detoken_questions = []
        for q in ordered_questions_token:
            string_q = "".join(
            [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in q]).strip() + '?'
            detoken_questions.append(string_q)


        for q in detoken_questions:
            print(q)

    # Visualize caption and attention of best sequence
    alphas = torch.FloatTensor(alphas)
    # visualize_att(args.img, seq1, alphas, rev_word_map, args.smooth, figure_name='output01.png')


def caption_image_beam_search(encoder, decoder, image_path, word_map, beam_size=7):
    """
    Reads an image and captions it with beam search.

    :param encoder: encoder model_VQG
    :param decoder: decoder model_VQG
    :param image_path: path to image
    :param word_map: word map
    :param beam_size: number of sequences to consider at each decode-step
    :return: caption, weights for visualization
    """

    k = beam_size
    vocab_size = len(word_map)

    # Read image and process
    img = imread(image_path)
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = imresize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    img = img / 255.
    img = torch.FloatTensor(img).to(device)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([normalize])
    image = transform(img)  # (3, 256, 256)

    # Encode
    image = image.unsqueeze(0)  # (1, 3, 256, 256)
    encoder_out = encoder(image)  # (1, enc_image_size, enc_image_size, encoder_dim)
    enc_image_size = encoder_out.size(1)
    encoder_dim = encoder_out.size(3)

    # Flatten encoding
    encoder_out = encoder_out.view(1, -1, encoder_dim)  # (1, num_pixels, encoder_dim)
    num_pixels = encoder_out.size(1)

    # We'll treat the problem as having a batch size of k
    encoder_out = encoder_out.expand(k, num_pixels, encoder_dim)  # (k, num_pixels, encoder_dim)

    # Tensor to store top k previous words at each step; now they're just <start>
    k_prev_words = torch.LongTensor([[word_map['<start>']]] * k).to(device)  # (k, 1)

    # Tensor to store top k sequences; now they're just <start>
    seqs = k_prev_words  # (k, 1)

    # Tensor to store top k sequences' scores; now they're just 0
    top_k_scores = torch.zeros(k, 1).to(device)  # (k, 1)

    # Tensor to store top k sequences' alphas; now they're just 1s
    seqs_alpha = torch.ones(k, 1, enc_image_size, enc_image_size).to(device)  # (k, 1, enc_image_size, enc_image_size)

    # Lists to store completed sequences, their alphas and scores
    complete_seqs = list()
    complete_seqs_alpha = list()
    complete_seqs_scores = list()

    # Start decoding
    step = 1
    h, c = decoder.init_hidden_state(encoder_out)

    # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
    while True:

        embeddings = decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

        awe, alpha = decoder.attention(encoder_out, h)  # (s, encoder_dim), (s, num_pixels)

        alpha = alpha.view(-1, enc_image_size, enc_image_size)  # (s, enc_image_size, enc_image_size)

        gate = decoder.sigmoid(decoder.f_beta(h))  # gating scalar, (s, encoder_dim)
        awe = gate * awe

        h, c = decoder.decode_step(torch.cat([embeddings, awe], dim=1), (h, c))  # (s, decoder_dim)

        scores = decoder.fc(h)  # (s, vocab_size)
        scores = F.log_softmax(scores, dim=1)

        # Add
        scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

        # For the first step, all k points will have the same scores (since same k previous words, h, c)
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            first_words = top_k_words.numpy().tolist()  # example: ['What', 'Is', 'How', 'Why', 'Where']

        else:
            # Unroll and find top scores, and their unrolled indices
            top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)
            # first_words.extend(top_k_words.numpy().tolist())
            # print(top_k_words)

        # Convert unrolled indices to actual indices of scores
        prev_word_inds = (top_k_words / vocab_size).type(torch.LongTensor)  # (s)
        next_word_inds = (top_k_words % vocab_size).type(torch.LongTensor)

        # Add new words to sequences, alphas
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)
        seqs_alpha = torch.cat([seqs_alpha[prev_word_inds], alpha[prev_word_inds].unsqueeze(1)],
                               dim=1)  # (s, step+1, enc_image_size, enc_image_size)

        # Which sequences are incomplete (didn't reach <end>)?
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != word_map['<end>']]
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        # Set aside complete sequences
        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())
            complete_seqs_alpha.extend(seqs_alpha[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
        k -= len(complete_inds)  # reduce beam length accordingly

        # Proceed with incomplete sequences
        if k == 0:
            break
        seqs = seqs[incomplete_inds]
        seqs_alpha = seqs_alpha[incomplete_inds]
        h = h[prev_word_inds[incomplete_inds]]
        c = c[prev_word_inds[incomplete_inds]]
        encoder_out = encoder_out[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

        # Break if things have been going on too long
        if step > 50:
            break
        step += 1

    # Return best sequence
    i = complete_seqs_scores.index(max(complete_seqs_scores))
    seq_1 = complete_seqs[i]
    alphas = complete_seqs_alpha[i]

    # Remove best score
    complete_seqs_scores[i] = float('-inf')

    # Return second best sequence
    complete_seqs_scores, seq_2 = return_different_seq(complete_seqs, complete_seqs_scores, seq_1)

    # TODO: compute similarity between questions
    # Compare with seq2 to get another different question
    try:
        complete_seqs_scores, seq_3 = return_different_seq(complete_seqs, complete_seqs_scores, seq_2)
    except:
        seq_3 = ''

    try:
        complete_seqs_scores, seq_4 = return_different_seq(complete_seqs, complete_seqs_scores, seq_3)
    except:
        seq_4 = ''

    try:
        complete_seqs_scores, seq_5 = return_different_seq(complete_seqs, complete_seqs_scores, seq_4)
    except:
        seq_5 = ''

    # All sequences
    all_seq = complete_seqs
    ordered_seq = [seq_1, seq_2, seq_3, seq_4, seq_5]

    return alphas, ordered_seq, all_seq


def return_different_seq(all_seq, all_scores, best_seq):
    """
    Compare first words of sequences to discard questions that are very similar
    :param all_seq: all generated questions
    :param all_scores: all scores of questions
    :param best_seq: sequence to compare with
    :return: all scores with repeated set to -inf and different question
    """
    s = []
    for ind, seq in enumerate(all_seq):
        if best_seq[1] == seq[1] and best_seq[2] == seq[2] and best_seq[3] == seq[3]:
            all_scores[ind] = float('-inf')

    # Return another sequence
    if max(all_scores) != float('-inf'):
        s = all_seq[all_scores.index(max(all_scores))]

    return all_scores, s


def visualize_att(image_path, seq, alphas, rev_word_map, smooth=True, figure_name=''):
    """
    Visualizes caption with weights at every word.

    Adapted from paper authors' repo: https://github.com/kelvinxu/arctic-captions/blob/master/alpha_visualization.ipynb

    :param image_path: path to image that has been captioned
    :param seq: question
    :param alphas: weights
    :param rev_word_map: reverse word mapping, i.e. ix2word
    :param smooth: smooth weights?
    """
    image = Image.open(image_path)
    image = image.resize([14 * 24, 14 * 24], Image.LANCZOS)

    words = [rev_word_map[ind] for ind in seq]

    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '%s' % (words[t]), color='black', backgroundcolor='white', fontsize=12)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8, multichannel=False)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.show()
    plt.savefig(figure_name)


def get_question(image):

    # path to model_VQG
    # model = '/work-nfs/mcaros/TFM/BEST_checkpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'
    word_map_path = './data/caption_data/WORDMAP_VQG_5_cap_per_img_2_min_word_freq.json'
    model_enc = '/work-nfs/mcaros/TFM/checkpoints/6w_2wf_VQG_model/13epoch/enc_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'
    model_dec = '/work-nfs/mcaros/TFM/checkpoints/6w_2wf_VQG_model/13epoch/dec_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'
    #
    # model_enc = '/work-nfs/mcaros/TFM/checkpoints/enc_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'
    # model_dec = '/work-nfs/mcaros/TFM/checkpoints/dec_statecheckpoint_VQG_5_cap_per_img_2_min_word_freq.pth.tar'

    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    rev_word_map = {v: k for k, v in word_map.items()}  # ix2word

    beam_size = 9

    encoder = Encoder()
    decoder = DecoderWithAttention(len(word_map))

    # Load model_VQG
    decoder.load_state_dict(torch.load(model_dec))
    encoder.load_state_dict(torch.load(model_enc))

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    encoder.eval()
    decoder.eval()

    # Encode, decode with attention and beam search
    alphas, ordered_seq, all_seq = caption_image_beam_search(encoder, decoder, image, word_map, beam_size)

    # Decode index to word
    ordered_questions_token = []
    for question in ordered_seq:
        q = [rev_word_map[ind] for ind in question[1:-1]]
        # Remove <unk>
        if '<unk>' in q:
            pass
        else:
            ordered_questions_token.append(q)

    # Detokenize question
    questions_list = []
    for q in ordered_questions_token:
        string_q = "".join(
            [" " + i if not i.startswith("'") and i not in string.punctuation else i for i in q]).strip() + '?'
        questions_list.append(string_q)

    questions = []
    print(questions_list)
    # Remove time questions
    for q in questions_list:

        append = True

        if q == 'What year was this photo taken?' or q == "What year was this picture taken?" or \
                q == 'How old is this picture?' or q == 'How old is this photo?' or q == 'What year is this picture?'\
                or q == 'What year is this photo?' or q =='?':
            continue

        for q2 in questions_list:
            if q != q2:
                similarity = get_jaccard_sim(q, q2)
                # print(similarity, q, q2)
                if similarity > 0.4:
                    append = False

        if append:
            questions.append(q)

    return questions


def get_jaccard_sim(str1, str2):
    a = set(str1.split())
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


if __name__ == '__main__':
    main()
