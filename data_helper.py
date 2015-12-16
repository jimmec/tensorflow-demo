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

def build_input_data(sentences, labels, vocabulary, vocab_inv):
    """
    Maps sentences to vectors based on vocabulary.
    Maps binary labels to 1 hot vectors, 0 -> [1,0], 1 -> [0,1]
    returns np.arrays
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array([[0,1] if label else [1,0] for label in labels])
    return x, y, vocabulary, vocab_inv

def read_pos_and_neg_data(header=True):
    with open('./data/raw_input_neg_500k.tsv', 'rt') as f:
        f.readline() if header else None
        lines = f.readlines()
        negatives = [(0, line.split('\t')[3]) for line in lines]
        print('num of negative examples: {}'.format(len(negatives)))
    with open('./data/raw_input_pos_500k.tsv', 'rt') as f:
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
    label_texts = np.array(list(label_texts))
    np.random.shuffle(label_texts)
    dev_size = int(p * label_texts.size)
    
    with open(path_template.format('dev'), 'wt') as df:
        for pair in label_texts[-dev_size:]:
            df.write(str.format('{}\t{}\n', pair[0], pair[1]))
    with open(path_template.format('train'), 'wt') as tf:
        for pair in label_texts[:-dev_size]:
            tf.write(str.format('{}\t{}\n', pair[0], pair[1]))

def preprocess_raw_inputs_and_save():
    labels, texts = zip(*read_pos_and_neg_data())
    sentences = [raw_text_to_sentence(text) for text in texts]
    normalized_sentences = normalize_sentence_length(sentences, max_length_words=68)
    vocab, inv_vocab = build_vocab(normalized_sentences)
    
    ts = datetime.datetime.now().strftime('%Y-%m-%d.%H%M%S')
    path_template = './data/{{}}_set.{}'.format(ts)
    with open('./data/vocab.pickle.{}'.format(ts), 'wb') as vocab_file:
        pickle.dump({'vocabulary': vocab, 'inv_vocabulary': inv_vocab}, vocab_file)
    write_dev_train_sets(zip(labels, [" ".join(sent) for sent in normalized_sentences]), path_template, p=0.1)

def load_input_data(example_path, vocab_path):
    with open(example_path, 'rt') as f:
        lines = f.readlines()
        labels = [line.split('\t')[0] for line in lines]
        texts = [line.split('\t')[1] for line in lines]
        sentences = map(raw_text_to_sentence, texts)
    with open(vocab_path, 'rb') as v:
        vocab_map = pickle.load(v)
    return build_input_data(sentences, labels, vocab_map['vocabulary'], vocab_map['inv_vocabulary'])

def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
