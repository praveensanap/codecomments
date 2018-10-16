import unicodedata
import os
import torch
import fastText
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

curr_dir = os.path.dirname(__file__)

model = fastText.load_model( os.path.join( curr_dir , "model/embeddings.bin"))

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeSentence(s):
    s = [i.lower().strip() for i in s.split(' ') if len(i.strip()) > 0]
    return ' '.join(s)


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    print(os.path.dirname(__file__))
    dataFile = os.path.join( curr_dir , 'data/%s-%s.txt' % (lang1, lang2))

    lines = open(dataFile, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeSentence(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    return len(p[0].split(' ')) < 50 and \
        len(p[1].split(' ')) < 50


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        if(len(pair) > 1): #FIXME irregular data
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence, one_hot = False):
    if one_hot:
        return [lang.word2index[word] for word in sentence.split(' ')]
    return [model.get_word_vector(word) for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence, one_hot=False):
    indexes = indexesFromSentence(lang, sentence, one_hot)
    if one_hot:
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1,1)
    indexes.append(model.get_word_vector('<EOS>'))
    return torch.tensor(np.asanyarray(indexes), dtype=torch.float32, device=device)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0],False)
    target_tensor = tensorFromSentence(output_lang, pair[1],True)
    return (input_tensor, target_tensor)