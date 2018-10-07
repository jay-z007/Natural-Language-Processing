import os
import pickle
import numpy as np
import yaml
import tensorflow as tf

hyperparams_config_path = './hyperparams.yaml'
config_name = 'exp4'

with open(hyperparams_config_path) as fp:
	hparams = yaml.load(fp)

hparams = hparams[config_name]
batch_size = hparams['batch_size']
embedding_size = hparams['embedding_size']  # Dimension of the embedding vector.
skip_window = hparams['skip_window']       # How many words to consider left and right.
num_skips = hparams['num_skips']         # How many times to reuse an input to generate a label.
lr = hparams['lr'] 
loss_model = hparams['loss_model']

experiment_name = "lr_%f__batch_%d__embed_%d__window_%d__nskip_%d"%(lr, batch_size, embedding_size, skip_window, num_skips)

model_path = './models/'
# loss_model = 'cross_entropy'

model_path = os.path.join(model_path, experiment_name)
model_filepath = os.path.join(model_path, 'word2vec_%s.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))
reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

valid_examples = ['first', 'american', 'would']
valid_size = 3
word_ids = [dictionary[word] for word in valid_examples]

graph = tf.Graph()
with tf.Session(graph=graph) as sess:
	valid_dataset = tf.constant(word_ids, dtype=tf.int32)

	norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
	normalized_embeddings = embeddings / norm

	valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
	
	similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

	sim = similarity.eval()

	for i in range(valid_size):
		valid_word = valid_examples[i]
		top_k = 20  # number of nearest neighbors
		nearest = (-sim[i, :]).argsort()[1:top_k + 1]
		log_str = "Nearest to %s:" % valid_word
		for k in range(top_k):
		  close_word = reverse_dictionary[nearest[k]]
		  log_str = "%s %s," % (log_str, close_word)
		print(log_str)