#!/bin/python

## This is used for cleaning the data before Brown Clustering
# import string
# from nltk.tokenize import TweetTokenizer
# tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
# lookup = {}

cluster_dict = {}

def preprocess_corpus(train_sents):
	"""Use the sentences to do whatever preprocessing you think is suitable,
	such as counts, keeping track of rare features/words to remove, matches to lexicons,
	loading files, and so on. Avoid doing any of this in token2features, since
	that will be called on every token of every sentence.

	Of course, this is an optional function.

	Note that you can also call token2features here to aggregate feature counts, etc.
	"""
	## Lookup is used for cleaning the text
	# lookup["'ll"] = "will"
	# lookup["'m"] = "am"
	# lookup["'s"] = "is"
	# lookup["'ve"] = "have"
	# lookup["'re"] = "are"
	# lookup["n't"] = "not"
	# lookup["im"] = "i am"
	# lookup["ill"] = "i will"
	# lookup["i'll"] = "i will"
	# lookup["i`ll"] = "i will"
	# lookup["id"] = "i would"
	# lookup["i'd"] = "i would"
	# lookup["lil"] = "little"

	filepath = "./data/paths_12"
	with open(filepath, 'rt') as f:
		for line in f.readlines():
			cluster, word, count = line.split()
			cluster_dict[word] = cluster


def token2features(sent, i, add_neighs = True):
	"""Compute the features of a token.

	All the features are boolean, i.e. they appear or they do not. For the token,
	you have to return a set of strings that represent the features that *fire*
	for the token. See the code below.

	The token is at position i, and the rest of the sentence is provided as well.
	Try to make this efficient, since it is called on every token.

	One thing to note is that it is only called once per token, i.e. we do not call
	this function in the inner loops of training. So if your training is slow, it's
	not because of how long it's taking to run this code. That said, if your number
	of features is quite large, that will cause slowdowns for sure.

	add_neighs is a parameter that allows us to use this function itself in order to
	recursively add the same features, as computed for the neighbors. Of course, we do
	not want to recurse on the neighbors again, and then it is set to False (see code).
	"""
	ftrs = []
	# bias
	ftrs.append("BIAS")
	# position features
	if i == 0:
		ftrs.append("SENT_BEGIN")
	if i == len(sent)-1:
		ftrs.append("SENT_END")

	# the word itself
	word = unicode(sent[i])
	ftrs.append("WORD=" + word)
	ftrs.append("LCASE=" + word.lower())
	# some features of the word
	if word.isalnum():
		ftrs.append("IS_ALNUM")
	if word.isnumeric():
		ftrs.append("IS_NUMERIC")
	if word.isdigit():
		ftrs.append("IS_DIGIT")
	if word.isupper():
		ftrs.append("IS_UPPER")
	if word.islower():
		ftrs.append("IS_LOWER")
	

	# is first char capital -> ease to identify nouns
	if word[0].isupper() and word[1:].islower():
		ftrs.append("IS_CAPITALIZED")

	word = word.lower()

	# if '-' in word:
	#     ftrs.append("IS_JOINT_WORD")

	# if 'http' in word:
	# 	ftrs.append("IS_URL")

	if word.startswith('#'):
		ftrs.append("IS_HASHTAG")        

	if word.startswith('@'):
		ftrs.append("IS_USERNAME")


	# -ing, -ed, -s, -ly, -ies, -ion, -ogy, -tion, -ity
	## word ending in -s
	if word.endswith('s'):
		ftrs.append("ENDS_WITH_S")

	# word ending in -ed
	if word.endswith('ed'):
		ftrs.append("ENDS_WITH_ED")
 
	## word ending in -ly
	if word.endswith('ly'):
		ftrs.append("ENDS_WITH_LY")

	## word ending in -ing
	if word.endswith('ing'):
		ftrs.append("ENDS_WITH_ING")

	## word ending in -ous 
	if word.endswith('ous'):
		ftrs.append("ENDS_WITH_OUS")

	## word ending in -ive
	if word.endswith('ive'):
		ftrs.append("ENDS_WITH_IVE")

	## word ending in -tion 
	if word.endswith('tion'):
		ftrs.append("ENDS_WITH_TION")

	## The commented features were tried, 
	## but they do not help to increase the accuracy
	# ## word ending in -ic
	# if word.endswith('ic'):
	#     ftrs.append("ENDS_WITH_IC")

	# ## word ending in -ful
	# if word.endswith('ful'):
	#     ftrs.append("ENDS_WITH_FUL")

	# ## word ending in -ies
	# if word.endswith('ies'):
	#     ftrs.append("ENDS_WITH_IES")

	# ## word ending in -ion
	# if word.endswith('ion'):
	#     ftrs.append("ENDS_WITH_ION")

	# ## word ending in -ity
	# if word.endswith('ity'):
	#     ftrs.append("ENDS_WITH_ITY")

	# ## word ending in -ogy
	# if word.endswith('ogy'):
	#     ftrs.append("ENDS_WITH_OGY") 

	# if word.endswith('ily'):
	#     ftrs.append("ENDS_WITH_ILY")

	# negating word
	negs = ["un", "non", "im", "in"]#, "il", "ir"] ## find more
	for neg in negs:
		if word.startswith(neg):
			ftrs.append("IS_NEG_WORD")

	## Only used for cleaning word before brown clustering
	# if word in lookup:
	# 	word = lookup[word]

	# for char in string.punctuation:
	# 	word = word.replace(char, '')

	# word = tknzr.tokenize(word)
	# if word:
	# 	word = word[0]
		
	if word in cluster_dict:
		ftrs.append("CLUSTER_{}".format(cluster_dict[word]))


	# previous/next word feats
	if add_neighs:
		if i > 0:
			for pf in token2features(sent, i-1, add_neighs = False):
				ftrs.append("PREV_" + pf)
		if i < len(sent)-1:
			for pf in token2features(sent, i+1, add_neighs = False):
				ftrs.append("NEXT_" + pf)

	# return it!
	return ftrs


if __name__ == "__main__":
	sents = [
	[ "I", "love", "food", ".", "I'll", 'illogical', 'impossible']
	]
	preprocess_corpus(sents)
	for sent in sents:
		for i in xrange(len(sent)):
			print sent[i], ":", token2features(sent, i)
