import os
import pickle
import numpy as np
import yaml

hyperparams_config_path = './hyperparams.yaml'
config_name = 'exp3'
# config_name = 'default'


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

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

questions_file_path = './word_analogy_test.txt'
out_file_path = 'word_analogy_dev_sample_predictions.txt'

similarity = lambda x,y : np.dot(x,y)/(np.linalg.norm(x) * np.linalg.norm(y))

with open(questions_file_path, 'rt') as input_file, open(out_file_path, 'wt') as output_file:
	for line in input_file:
		line = line.split('\n')[0]
		examples, choices = line.split('||')
		examples = [tuple(e[1:-1].split(':')) for e in examples.split(',')]
		choices = [tuple(c[1:-1].split(':')) for c in choices.split(',')]
		# print (choices)
		# break
		diffs = []
		
		for word1, word2 in examples:
			vec1 = embeddings[dictionary[word1]]
			vec2 = embeddings[dictionary[word2]]

			diffs.append(vec2-vec1)

		avg_diff = np.array(diffs).mean(axis=0)

		similar = []

		for word1, word2 in choices:
			vec1 = embeddings[dictionary[word1]]
			vec2 = embeddings[dictionary[word2]]

			similar.append(similarity(avg_diff, vec2-vec1))

		least_illustrative = np.argmin(similar)
		most_illustrative = np.argmax(similar)

		output_line = line.split('||')[1].split(',')
		output_line.append(output_line[least_illustrative])
		output_line.append(output_line[most_illustrative])

		output_file.write(' '.join(output_line)+"\n")

		
"""
Run 
./score_maxdiff.pl word_analogy_dev_mturk_answers.txt word_analogy_dev_sample_predictions.txt score.txt

"""