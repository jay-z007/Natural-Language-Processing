from nltk.tokenize import TweetTokenizer
from data import read_twitter
import string

"""
## things to do for cleaning:
- I 'm -> Im, I 've to Ive, [do, does] n't to [do, does]nt, It's to Its, you 're to youre
- lowercase every thing
- remove RT : @handle
- remove punctuations
- remove URL's, hashtags and user handles [removing handles/mentions can be grammatically incorrect]
- consider removing stop words as well using nltk.corpus
- consider replacing the word with its stem [only after everything is done]

"""
lookup = {}
lookup["'ll"] = "will"
lookup["'m"] = "am"
lookup["'s"] = "is"
lookup["'ve"] = "have"
lookup["'re"] = "are"
lookup["n't"] = "not"
lookup["im"] = "i am"
lookup["ill"] = "i will"
lookup["i'll"] = "i will"
lookup["i`ll"] = "i will"
lookup["id"] = "i would"
lookup["i'd"] = "i would"
lookup["lil"] = "little"
# lookup["gonna"] = "going to"

tags_to_remove = set(["X", ".", "PRT"])


def clean_word(word, tag, lookup, tags_to_remove):
	word = word.lower()
	# if tag in tags_to_remove:
	# 	return None

	if word in lookup:
		word = lookup[word]

	for char in string.punctuation:
		word = word.replace(char, '')

	return word


if __name__ == '__main__':

	file_path = "./data/clean_sents.txt"

	data = read_twitter()
	tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)

	with open(file_path, 'wt') as f:
		for sent, label in zip(data.train_sents, data.train_labels):
			
			new_sent = []

			for word, tag in zip(sent, label):
				word = clean_word(word, tag, lookup, tags_to_remove)	
				if word:
					new_sent.append(word)

			new_sent = tknzr.tokenize(' '.join(new_sent))
			f.write(' '.join(new_sent)+"\n")
