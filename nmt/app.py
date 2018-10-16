import torch

from nmt.model import EncoderRNN,AttnDecoderRNN
from nmt.evaluate import evaluate, evaluateRandomly

hidden_size = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = '/nmt/model/code-comments.pt'

checkpoint = torch.load(PATH)

input_lang = checkpoint['input_lang']
output_lang = checkpoint['output_lang']
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
encoder.load_state_dict(checkpoint['encoder'])
decoder.load_state_dict(checkpoint['decoder'])
encoder.eval()
decoder.eval()

input = 'public void add ( int a , int b ) { return a + b ; }'

output, _ = evaluate(encoder, decoder,input, input_lang, output_lang)

print(' '.join(output))


