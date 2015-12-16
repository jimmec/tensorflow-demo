import numpy as np
import datetime
import itertools
import pickle
from collections import Counter
import re
"""
Code adapted from https://github.com/dennybritz/cnn-text-classification-tf
"""

def clean_str(string):
    """
    Tokenization/string cleaning for input.
    Originally taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = tokenize(string)
    return string.strip().lower()

def tokenize(string):
    """
    Tokenize English strings
    """
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " ( ", string) 
    string = re.sub(r"\)", " ) ", string) 
    string = re.sub(r"\?", " ? ", string) 
    string = re.sub(r"\s{2,}", " ", string) 
    return string
    
def normalize_sentence_length(sentences, padding_word='<PAD/>', max_length_words = None):
    """
    Normalizes lengths to max_length. max_length = None, normalizes to max sentence length.
    Pads shorter sentences using the padding_word. 
    Cuts longer sentences at max_length number of words.
    
    sentences - list of sentence where each sentence is a list of words. eg. [['foo','bar'], ['fo2','bar2']]
    """
    max_length_words = max_length_words if max_length_words is not None else max(len(x) for x in sentences)
    norm_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = max_length_words - len(sentence)
        padded_sentence = sentence + [padding_word] * num_padding
        chopped_sentence = padded_sentence[:max_length_words]
        norm_sentences.append(chopped_sentence)
    return norm_sentences

def raw_text_to_sentence(text):
    return clean_str(text).split(" ")

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    print('building vocab...')
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentences to vectors based on vocabulary.
    Maps binary labels to 1 hot vectors, 0 -> [1,0], 1 -> [0,1]
    returns np.arrays
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array([[0,1] if label else [1,0] for label in labels])

def read_pos_and_neg_data(header=True):
    with open('./data/raw_input_neg.tsv', 'rt') as f:
        f.readline() if header else None
        lines = f.readlines()
        negatives = [(0, line.split('\t')[3]) for line in lines]
        print('num of negative examples: {}'.format(len(negatives)))
    with open('./data/raw_input_pos.tsv', 'rt') as f:
        f.readline() if header else None
        lines = f.readlines()
        positives = [(1, line) for line in lines]
        print('num of positive examples: {}'.format(len(positives)))
    return positives + negatives

def write_dev_train_sets(label_texts, path_template, p=0.1):
    """
    Outputs a dev data set using apprx p percent of the list of label and text tuples label_texts
    Rest goes to the training set
    """
    print('saving dev and training data sets...')
    # sample p% for dev use 
    dev_mask = np.random.rand(len(label_texts)) <= p
    train_mask = np.array([not d for d in dev_mask])
    full = np.array(label_texts)
    dev = full[dev_mask]
    train = full[train_mask]
    
    with open(path_template.format('dev'), 'wt') as df:
        for pair in dev:
            df.write(str.format('{}\t{}\n', pair[0], pair[1]))
    with open(path_template.format('train'), 'wt') as tf:
        for pair in train:
            tf.write(str.format('{}\t{}\n', pair[0], pair[1]))

def preprocess_raw_inputs_and_save():
    labels, texts = zip(*read_pos_and_neg_data())
    sentences = [raw_text_to_sentence(text) for text in texts]
    normalized_sentences = normalize_sentence_length(sentences)
    vocab, inv_vocab = build_vocab(normalized_sentences)
    
    ts = datetime.datetime.now().strftime('%Y-%m-%d.%H%M%S')
    path_template = './data/{{}}_set.{}'.format(ts)
    with open('./data/vocab.pickle.{}'.format(ts), 'wt') as vocab_file:
        cPickle.dump({'vocabulary': vocab, 'inv_vocabulary': inv_vocab}, vocab_file)
    write_dev_train_sets(zip(labels, [" ".join(sent) for sent in normalized_sentences]), path_template, p=0.1)

preprocess_raw_inputs_and_save()
