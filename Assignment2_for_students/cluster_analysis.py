import math
from collections import defaultdict

class Cluster(object):
	"""docstring for Cluster"""
	def __init__(self, n):
		"""
			input:
			n : number of bits to consider for making clusters
		"""
		super(Cluster, self).__init__()
		self.n = n
		
	def parse(self, file_path):
		"""
			Parse the input file and generate a dict to hold the word, bit_string and count
		"""
		# self.data = defaultdict(tuple)
		data = {}

		with open(file_path, 'rt') as f:
			for line in f.readlines():
				word, bit_string, count = line.split()
				data[bit_string] = (word, count)
		
		# print(data)
		self.brown_clusters = data
		print("Successfully parsed {}".format(file_path))

	def analyse(self):
		self.merged_clusters = defaultdict(list)
		shortest_bit_string = min(self.brown_clusters.keys(), key=lambda x: len(x))
		print (shortest_bit_string)


		for k, v in self.brown_clusters.items():
			if len(k) >= self.n:
				k = k[:self.n]

			self.merged_clusters[k].append(v)


		for k, v in self.merged_clusters.items():
			print(k, v)
			

if __name__ == '__main__':
	obj = Cluster(8)
	obj.parse('./data/train_sent_clusters.txt')
	obj.analyse()


