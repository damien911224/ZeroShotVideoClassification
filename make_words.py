import os
import json
from gensim.models import KeyedVectors as Word2Vec
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

wv_model = Word2Vec.load('./assets/GoogleNewsAdded', mmap='r')

caption_folder = "/mnt/hdd1/captions"
with open(os.path.join(caption_folder, "COCO", "image_captions.json"), "r") as fp:
    image_captions = json.load(fp)
with open(os.path.join(caption_folder, "ActivityNet", "video_captions.json"), "r") as fp:
    video_captions = json.load(fp)

nouns = list()
adjectives = list()
verbs = list()
adverbs = list()
for caption in image_captions:
    for w in word_tokenize(caption):
        analysis = wn.synsets(w)
        if any([a.pos() in ['n'] for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'n')
            nouns.append(w)

        elif any([a.pos() in ['a', 's'] for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'a')
            adjectives.append(w)
            print(w)

        elif any([a.pos() in ['v'] for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'v')
            verbs.append(w)

        elif any([a.pos() in ['r'] for a in analysis]):
            w = WordNetLemmatizer().lemmatize(w, 'r')
            adverbs.append(w)