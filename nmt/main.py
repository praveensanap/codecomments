from __future__ import unicode_literals, print_function, division

import random
import torch
import matplotlib.pyplot as plt

from nmt.data import prepareData
from nmt.model import EncoderRNN, AttnDecoderRNN
from nmt.train import trainIters
from nmt.evaluate import evaluateRandomly,evaluate,evaluateAndShowAttention

plt.switch_backend('TKAgg')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10



def run(model1, model2):

    #HYPERPARAMETERS
    hidden_size = 300
    dropout_p = 0.1
    n_iters = 2000
    print_every = 100
    teacher_forcing_ratio = 0
    learning_rate = 0.0001
    #Prepare Data
    input_lang, output_lang, pairs = prepareData('code', 'comment')


    # Define Model
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p).to(device)

    #Train
    trainIters(
        encoder=encoder1,
        decoder=attn_decoder1,
        pairs=pairs,
        input_lang = input_lang,
        output_lang=output_lang,
        n_iters=n_iters,
        print_every=print_every,
        learning_rate=learning_rate,
        teacher_forcing_ratio=teacher_forcing_ratio)

    PATH = '/Users/praveensanap/dev/nlp/codecomments/codecomments-src/nmt/model/code-comments.pt'

    # Save Model, Parameters
    torch.save(
        {'encoder' : encoder1.state_dict(),
         'decoder' : attn_decoder1.state_dict(),
         'input_lang' : input_lang,
         'output_lang' : output_lang
         }, PATH)

    # Evaluate
    evaluateRandomly(encoder1, attn_decoder1,input_lang, output_lang, pairs)
    #
    # output_words, attentions = evaluate(encoder1, attn_decoder1, "je suis trop froid .",input_lang, output_lang,)
    #
    # plt.matshow(attentions.numpy())
    #
    # evaluateAndShowAttention(encoder1, attn_decoder1, "elle a cinq ans de moins que moi .",input_lang, output_lang)
    # evaluateAndShowAttention(encoder1, attn_decoder1,"elle est trop petit .",input_lang, output_lang)
    # evaluateAndShowAttention(encoder1, attn_decoder1,"je ne crains pas de mourir .",input_lang, output_lang)
    # evaluateAndShowAttention(encoder1, attn_decoder1,"c est un jeune directeur plein de talent .",input_lang, output_lang)

run('./model/cc_encoder.pt' , './model/cc_decoder.pt')