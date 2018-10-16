import torch

from nmt.data import prepareData

PATH = '/Users/praveensanap/dev/nlp/codecomments/codecomments-src/nmt/model/code-comments.pt'

checkpoint = torch.load(PATH)

input_lang = checkpoint['input_lang']
output_lang = checkpoint['output_lang']

input_lang, output_lang, pairs = prepareData('code', 'comment')

input_vocab = sorted(input_lang.word2count, key=input_lang.word2count.get ,reverse=True )
output_vocab = sorted(output_lang.word2count, key=output_lang.word2count.get, reverse=True)

