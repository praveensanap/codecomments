import javalang
import re
import time
import os
import random

DATA_PATH = '/Users/praveensanap/dev/nlp/codecomments/codecomments-data/java_projects/'
file = '/Users/praveensanap/dev/nlp/codecomments/codecomments-data/java_projects/guava-libraries/guava/'
pairsFile = './nmt/data/code-comment.txt'
corpusFile = './nmt/data/corpus.txt'
metadata = './nmt/data/metadata'

prjs = os.listdir(DATA_PATH)

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def preproc(doc):
    return ' '.join([normalizeString(i) for i in doc.strip().replace('\n',' ').split(' ') if normalizeString(i)]).strip()


def parse(tokens):
    ptr = 0

    modifiers =['abstract', 'default', 'final', 'native', 'private',
                  'protected', 'public', 'static', 'strictfp', 'synchronized',
                  'transient', 'volatile']
    pairs = []

    while ptr < len(tokens):
        curr= tokens[ptr]
        if curr.value in modifiers:
            doc = curr.javadoc
            method = [curr.value]
            stack = list()
            while ptr < len(tokens) - 1:
                ptr += 1
                curr = tokens[ptr]
                method.append(curr.value)
                if curr.value == 'class':
                    method = list()
                    break
                if curr.value == '{':
                    stack.append('{')
                    break

            while len(stack) > 0:
                ptr += 1
                curr = tokens[ptr]
                method.append(curr.value)
                if curr.value == '{':
                    stack.append('}')
                if curr.value == '}':
                    stack.pop()

            if len(method) > 0 and doc and len(doc) > 0:
                pairs.append((' '.join(method), preproc(doc)))
            method = list()
            doc = None
        ptr += 1
    return pairs


from os import walk

def jwalk(file):
    for i in walk(file):
        for j in i[2]:
            if 'java' in j:
                jf = i[0] + "/" + j
                yield jf



def embeddingCorpus(tokens):
    string = []
    for i in tokens:
        if i.javadoc:
            string.append(preproc(i.javadoc))
        string.append(i.value)
    return ' '.join(string)





def load(prj):
    with open(pairsFile, 'a') as p , open(corpusFile, 'a') as c:
        t = time.time()
        count = 0
        for i in jwalk(prj):
            count+=1
            try :
                with open(i , encoding='utf-8') as j:
                    code = j.read()
                    tokens = list(javalang.tokenizer.tokenize(code))
                    for k in parse(tokens):
                        pp = '\t'.join(k)
                        try:

                            p.write(pp)
                            p.write('\n')
                        except:
                            #FIXME ignoring encoding problem
                            pass
                    e = embeddingCorpus(tokens)
                    try:
                        c.write(e)
                        c.write(' ')
                    except:
                        # FIXME ignoring encoding problem
                        pass

            except:
                pass
        print("Processes ", count, " files in ",time.time() - t, " seconds")



def fetch():
    with open(pairsFile,'w') as p, open(corpusFile,'w') as c:
        p.write('')
        c.write('')
    chosen = []
    while len(chosen) < 100:
        c  = random.choice(prjs)
        if not c in chosen:
            chosen.append(c)
            load(DATA_PATH + c)
    with open(metadata, 'w') as m:
        m.write('\n'.join(chosen))

fetch()






