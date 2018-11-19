import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            ...
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """
            ## Freezing the embeddings
            # self.embeddings = tf.Variable(embedding_array, dtype=tf.float32, trainable=False)
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32)

            n_transitions = parsing_system.numTransitions()

            self.train_inputs = tf.placeholder(tf.int32, shape=(Config.batch_size, Config.n_Tokens))
            self.train_labels = tf.placeholder(tf.float32, shape=(Config.batch_size, n_transitions))
            self.test_inputs = tf.placeholder(tf.int32, shape=(48,))

            ## lookup from embedding array
            train_embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            train_embed = tf.reshape(train_embed, [Config.batch_size, -1])


            ## DO NOT USE THIS FOLLOWING COMMENTED BLOCK
            ## generate weights and biases (Default) [*** Use the generalized weights]
            # weights_input = tf.Variable(tf.random.truncated_normal(shape=(Config.embedding_size*Config.n_Tokens , Config.hidden_size), stddev=0.1))
            # biases_input = tf.Variable(tf.random.truncated_normal(shape=(1, Config.hidden_size), stddev=0.1))
            # weights_output = tf.Variable(tf.random.truncated_normal(shape=(Config.hidden_size, n_transitions), stddev=0.1))

            ## Generalized weight and bias init for sequential hidden layers
            shapes = [Config.embedding_size*Config.n_Tokens] + Config.hidden_size + [n_transitions]
            weights = [tf.Variable(tf.random.truncated_normal((shape1, shape2), stddev=0.1)) for shape1, shape2 in zip(shapes, shapes[1:])]
            biases = [tf.Variable(tf.random.truncated_normal((1, shape), stddev=0.1)) for shape in Config.hidden_size]

            ## Weights init for Parallel hidden layers
            # weights_word = tf.Variable(tf.random.truncated_normal((Config.embedding_size*18, Config.hidden_size[0]), stddev=0.1))
            # weights_pos = tf.Variable(tf.random.truncated_normal((Config.embedding_size*18, Config.hidden_size[0]), stddev=0.1))
            # weights_dep = tf.Variable(tf.random.truncated_normal((Config.embedding_size*12, Config.hidden_size[0]), stddev=0.1))
            # weights_output = tf.Variable(tf.random.truncated_normal((Config.hidden_size[0]*3, n_transitions), stddev=0.1))
            # weights = [weights_word, weights_pos, weights_dep, weights_output]
            # biases = [tf.Variable(tf.random.truncated_normal((1, Config.hidden_size[0]), stddev=0.1))]*3
    

            ## Generate mask for ignoring the transitions with -1 label
            mask = tf.cast(self.train_labels >= 0, tf.float32)

            ## forward pass
            # self.prediction = self.forward_pass(train_embed, weights_input, biases_input, weights_output)
            self.prediction = self.forward_pass(train_embed, weights, biases)
            
            pred = tf.math.multiply(self.prediction, mask)
            labels = tf.argmax(self.train_labels, axis=1)
            
            ## Loss
            ## For regularization: creating a combined list of weights and biases and 
            ## then sum over the l2 loss of each item in that list
            regularization = Config.lam * sum(tf.nn.l2_loss(w) for w in weights+biases) 
            cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits=pred, labels=labels)
            self.loss = cross_entropy + regularization

            ## Trying Different Optimizers
            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            # optimizer = tf.train.AdagradOptimizer(Config.learning_rate)
            # optimizer = tf.train.AdamOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            # self.app = optimizer.apply_gradients(grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            # self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)
            self.test_pred = self.forward_pass(test_embed, weights, biases)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


    def forward_pass(self, embed, weights, biases):
#     def forward_pass(self, embed, weights_input, biases_input, weights_output):
        """

        :param embed: batch_size, feature_size*embedding_size
        :param weights: feature_size*embedding_size, hidden_size
        :param biases: hidden_size
        :return:
        """
        """
        =======================================================

        Implement the forwrad pass described in
        "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

        =======================================================
        """
        h = embed
        
        for i in range(len(weights)-1):
            h = tf.matmul(h, weights[i]) + biases[i]
            h = tf.pow(h, 3)
#             h = tf.sigmoid(h)
#             h = tf.tanh(h)
#             h = tf.nn.relu(h)
        return tf.matmul(h, weights[-1])



##         Uncomment this for parallel hidden layers
#         word = embed[::, :18*Config.embedding_size]
#         pos = embed[::, 18*Config.embedding_size: 36*Config.embedding_size]
#         dep = embed[::, 36*Config.embedding_size:]
#         h_word = tf.pow(tf.matmul(word, weights[0]) + biases[0], 3)
#         h_pos = tf.pow(tf.matmul(pos, weights[1]) + biases[1], 3)
#         h_dep = tf.pow(tf.matmul(dep, weights[2]) + biases[2], 3)
        
#         h = tf.concat([h_word, h_pos, h_dep], axis=1)
#         return tf.matmul(h, weights[-1])
        
        


def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]


def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    features = []

    ## Token indices
    s1 = c.getStack(0)
    s2 = c.getStack(1)
    s3 = c.getStack(2)
    b1 = c.getBuffer(0)
    b2 = c.getBuffer(1)
    b3 = c.getBuffer(2)

    lc1_s1 = c.getLeftChild(s1, 1)
    rc1_s1 = c.getRightChild(s1, 1)
    lc2_s1 = c.getLeftChild(s1, 2)
    rc2_s1 = c.getRightChild(s1, 2)

    lc1_s2 = c.getLeftChild(s2, 1)
    rc1_s2 = c.getRightChild(s2, 1)
    lc2_s2 = c.getLeftChild(s2, 2)
    rc2_s2 = c.getRightChild(s2, 2)

    lc1_lc1_s1 = c.getLeftChild(lc1_s1, 1)
    rc1_rc1_s1 = c.getRightChild(rc1_s1, 1)

    lc1_lc1_s2 = c.getLeftChild(lc1_s2, 1)
    rc1_rc1_s2 = c.getRightChild(rc1_s2, 1)

    ## Word IDs
    word_ids = [
        getWordID(c.getWord(s1)),
        getWordID(c.getWord(s2)),
        getWordID(c.getWord(s3)),
        getWordID(c.getWord(b1)),
        getWordID(c.getWord(b2)),
        getWordID(c.getWord(b3)),
        getWordID(c.getWord(lc1_s1)),
        getWordID(c.getWord(rc1_s1)),
        getWordID(c.getWord(lc2_s1)),
        getWordID(c.getWord(rc2_s1)),
        getWordID(c.getWord(lc1_s2)),
        getWordID(c.getWord(rc1_s2)),
        getWordID(c.getWord(lc2_s2)),
        getWordID(c.getWord(rc2_s2)),
        getWordID(c.getWord(lc1_lc1_s1)),
        getWordID(c.getWord(rc1_rc1_s1)),
        getWordID(c.getWord(lc1_lc1_s2)),
        getWordID(c.getWord(rc1_rc1_s2))
    ]
    
    ## POS IDs
    pos_ids = [
        getPosID(c.getPOS(s1)),
        getPosID(c.getPOS(s2)),
        getPosID(c.getPOS(s3)),
        getPosID(c.getPOS(b1)),
        getPosID(c.getPOS(b2)),
        getPosID(c.getPOS(b3)),
        getPosID(c.getPOS(lc1_s1)),
        getPosID(c.getPOS(rc1_s1)),
        getPosID(c.getPOS(lc2_s1)),
        getPosID(c.getPOS(rc2_s1)),
        getPosID(c.getPOS(lc1_s2)),
        getPosID(c.getPOS(rc1_s2)),
        getPosID(c.getPOS(lc2_s2)),
        getPosID(c.getPOS(rc2_s2)),
        getPosID(c.getPOS(lc1_lc1_s1)),
        getPosID(c.getPOS(rc1_rc1_s1)),
        getPosID(c.getPOS(lc1_lc1_s2)),
        getPosID(c.getPOS(rc1_rc1_s2))
    ]

    ## Labels IDs
    label_ids = [
        getLabelID(c.getLabel(lc1_s1)),
        getLabelID(c.getLabel(rc1_s1)),
        getLabelID(c.getLabel(lc2_s1)),
        getLabelID(c.getLabel(rc2_s1)),
        getLabelID(c.getLabel(lc1_s2)),
        getLabelID(c.getLabel(rc1_s2)),
        getLabelID(c.getLabel(lc2_s2)),
        getLabelID(c.getLabel(rc2_s2)),
        getLabelID(c.getLabel(lc1_lc1_s1)),
        getLabelID(c.getLabel(rc1_rc1_s1)),
        getLabelID(c.getLabel(lc1_lc1_s2)),
        getLabelID(c.getLabel(rc1_rc1_s2))
    ]

    features.extend(word_ids)
    features.extend(pos_ids)
    features.extend(label_ids)

    return features



def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)

            # if c.tree.equal(trees[i]):
            #     print("Apply function is working Fine")

    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))
    
    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()    
    foundEmbed = 0
    
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    
    	    ## This following line is for creating a fixed bit vector for POS and Labels
            # embedding_array[i][i%Config.embedding_size] = 1
            
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

