# ./fasttext skipgram -input \
# /Users/praveensanap/dev/nlp/codecomments/codecomments-src/nmt/data/corpus.txt \
# -output \
# /Users/praveensanap/dev/nlp/codecomments/codecomments-src/nmt/model/embeddings \
# -minn 2 -maxn 5 -dim 300



import fastText

path = '/Users/praveensanap/dev/nlp/codecomments/codecomments-src/nmt/model/embeddings.vec'

model = fastText.load_model(path)



