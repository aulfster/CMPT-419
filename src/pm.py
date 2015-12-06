#! /usr/bin/python

import sys, os, optparse, logging, time
from collections import defaultdict
from pyspark import SparkContext


# Debug output flag
debug = False

START = '*'
STOP  = 'STOP'

TRIGRAM = 'TRIGRAM'
TAG = 'TAG'
#SUFFIX = 'SUFF'

class Trainer:
    def __init__(self):
        self.tags = set()
        self.f = set()
        self.v = {}
        self.x = []
        self.y = []

    def read_training_data(self, training_file):
        """
        """
        if debug: sys.stdout.write("Reading training data...\n")

        file = open(training_file, 'r')
        # file.seek(0) sets the file's current reading position at offset 0
        file.seek(0)
        sentence = []
        tags = []
        for line in file:
            #String.strip([chars]);
            #returns a copy of the string in which all chars have been stripped from the beginning and the end of the string.
            #e.g str = "0000000this is string example....wow!!!0000000"; 
            #    print str.strip( '0' )
            # answer: this is string example....wow!!!
            line = line.strip()
            if line: # Non-empty line
                #method split() returns a list of all the words in the string, using a delimiter (splits on all whitespace if left unspecified)
                token = line.split()
                word = token[0]
                tag  = token[3]
                #appends the word at the end of the list
                sentence.append(word)
                #appends the tag at the end of the list
                tags.append(tag)
            else: # End of sentence reached
                #converts the sentence list to a tuple and then appends that tuple to the end of the self.x list
                self.x.append(tuple(sentence))
                self.y.append(tuple(tags))
                #set object is an unordered collection of items and don't have duplicate items
                #set.update(other, ...)
                #Update the set, adding elements from all others
                # so self.tags.update(tags) keeps only the distinct # of tags for that sentence, that is, set([0,I-Gene]) for all sentences 
                self.tags.update(tags)
                sentence = []
                tags = []
        file.close()

    def read_weights(self, weight_file):
        """
        Read the previously determined weight vector
        features and values from the input file.
        """
        if debug: sys.stdout.write("Reading weight vector...\n")

        file = open(weight_file, 'r')
        file.seek(0)
        for line in file:
            token = line.split()
            feature = token[0]
            weight  = float(token[1])
            self.f.add(feature)
            self.v[feature] = weight

            # Add the tags from any trigram features to the list of tags
            token = feature.split(':')
            if token[0] == TRIGRAM:
                self.tags.update(token[1:])
        file.close()

        # Remove the start and stop symbols from the list of tags
        self.tags.remove(START)
        self.tags.remove(STOP)

    def feature_vector(self, history, tag):
        """
        Compute the feature vector g(h, t) for the given history/tag pair.
        """
        # Store the feature vector as a map from feature strings to counts.
        g = {}

        # Generate the possible features to search
        f = set()

        # Trigram feature
        f.add(TRIGRAM+':'+history['t2']+':'+history['t1']+':'+tag)

        # Tag feature (if not beyond the end of the sentence)
        if history['i'] <= len(history['x']):
            # history['x'][history['i'] = ['i','am','wasifa'][1-1] = ['i','am','wasifa'][0] 
            # so we are getting that word (item in a list) from the x list (each sentence) using the index, [history['i'] - 1]
            # so index [1 - 1] = index 0 would return word 'i' from the list sentence x.
            f.add(TAG+':'+history['x'][history['i'] - 1]+':'+tag)

        # Suffix features (if not beyond the end of the sentence)
        #if history['i'] <= len(history['x']):
        #    # for j=1:3-1
        #    for j in range(1,3):
        #        word = history['x'][history['i'] - 1]
        #        if j <= len(word):
        #            # word[-j:] where -j when j=1 refers to the last element 
        #            f.add(SUFFIX+':'+word[-j:]+':'+str(j)+':'+tag)

        # Check each feature
        for feature in f:
            token = feature.split(':')

            # Calculate trigram features
            if token[0] == TRIGRAM:
                s = token[1]
                u = token[2]
                v = token[3]
                if history['t2'] == s and history['t1'] == u and tag == v:
                    # g is a dictionary storing the number of times that each feature occured (value) using as index (key) the feature string, e.g 'TRIGRAM:'*':'*':0'
                    g[feature] = 1

            # Calculate tag features
            if token[0] == TAG:
                # Handle cases where ':' appears as part of the word
                # join() method returns a string by joining a sequence of elements using a separator
                # ("-").join(("a", "b", "c")) would return string a-b-c
                r = ':'.join(token[1:-1])
                u = token[-1]
                if history['i'] <= len(history['x']):
                    w = history['x'][history['i'] - 1]
                else:
                    w = ''
                if w == r and tag == u:
                    g[feature] = 1


            ## Calculate suffix features
            #if token[0] == SUFFIX:
            #    # Handle cases where ':' appears as part of the word
            #    u = ':'.join(token[1:-2])
            #    j = int(token[-2])
            #    v = token[-1]
            #    if history['i'] <= len(history['x']):
            #        w = history['x'][history['i'] - 1]
            #    else:
            #        w = ''
            #    if w[-j:] == u and tag == v:
            #        g[feature] = 1

        return g

    def inner_product(self, g):
        """
        Compute the inner product v.g(h, t)
        """
        # The method get() returns a value for the given key. If key is not available then returns default value None or 0 for below case.
        # g.iteritems() method returns an iterator over the dictionary g's (key, value) pairs
        return sum((self.v.get(key, 0) * value for key, value in g.iteritems()))

    def reset_weights(self):
        """
        Reset all the weights to zero in the weight vector.
        """
        for feature in self.v:
            self.v[feature] = 0

    def viterbi_algorithm(self, x):
        """
        Run the GLM form of the Viterbi algorithm on the input sentence (x)
        using the previously determined feature (f) and weight (v) vectors.
        """
        #total number of words in sentence x
        n = len(x)

        # Initialise pi and bp
        pi = {}
        bp = {}
        # pi{key:value} where key is a tuple, that is, (k th position, tags in k-1 position, tags in kth position)
        # so, pi[(0, '*', '*')] means pi[key]=pi[(k,T,U)] where k=0,T='*',U='*'
        pi[(0, '*', '*')] = 0

        # for k=1:n+1-1 in MATLAB style
        # that is, for each word, k, in the sentence
        for k in range(1, n+1):

            if k == 1:
                # T is possible tags of second previous word in position k-2
                T = {'*'}
                # U is possible tags of first previous word in position k-1
                U = {'*'}
                # S is possible tags of the current word in position k
                S = self.tags
            elif k == 2:
                T = {'*'}
                U = self.tags
                S = self.tags
            else:
                T = self.tags
                U = self.tags
                S = self.tags

            for u in U:
                for s in S:
                    pi[(k, u, s)], bp[(k, u, s)] = max(((pi[(k-1, t, u)] + self.inner_product(self.feature_vector(dict(t2=t, t1=u, x=x, i=k), s)), t) for t in T))

        # Store the tag sequence as an array
        tag = ['']*(n+1)

        # Calculate the tag sequence by following the back pointers
        prob, tag[n-1], tag[n] = max((max(((pi[(n, u, s)] + self.inner_product(self.feature_vector(dict(t2=u, t1=s, x=x, i=n+1), STOP)), u, s) for s in S)) for u in U))
        for k in range(n-2, 0, -1):
            tag[k] = bp[(k+2, tag[k+1], tag[k+2])]
        tag = tag[1:]

        # Return the probability and tag sequence
        return prob, tag

    def perceptron_algorithm(self, iterations):
        """
        Run the perceptron algorithm to estimate (or improve) the weight vector.
        """
        self.reset_weights()


        # range(iterations) = [0,1,2,3,4]
        # for iteration=0:4 in MATLAB style
        for iteration in range(iterations):
            #if debug: sys.stdout.write("Perceptron algorithm iteration %d...\n" % (iteration+1))

            # Loop over each sentence/tag pair (xi, yi) in the training data

            # len(self.x) returns total number of sentences in training data
            # range(len(self.x)) = [0,1,2...,total number of sentences in training data-1]
            # for i=0:total number of sentences in training data-1 in MATLAB style
            for i in range(len(self.x)):
                #list(self.x[i]) converts the given tuple, self.x[i],that is, each sentence, into a list.
                x = list(self.x[i])
                #list(self.y[i]) converts the given sentence tuple, self.y[i], that is, tags of each sentence, into a list.
                y = list(self.y[i])

                # Find the best tagging sequence using the Viterbi algorithm
                prob, z = self.viterbi_algorithm(x)

                # Check if the tags match the gold standard
                if z != y:

                    # Compute the gold tagging feature vector f(x, y)
                    # and the best tagging feature vector f(x, z)
                    fy = {}
                    fz = {}

                    # Add the start and stop symbols to the tagging sequences
                    x = x + [STOP]
                    y = [START, START] + y + [STOP]
                    z = [START, START] + z + [STOP]

                    # Loop over all words in the sentence
                    for k in range(len(x)):

                        # Calculate the features for the next gold tag (y_k)
                        f = self.feature_vector(dict(t2=y[k], t1=y[k+1], x=x, i=k+1), y[k+2])

                        # Add the features to the gold tagging feature vector
                        for feature in f:
                            # This method returns the key value available in the dictionary and if given key is not available then it will return provided default value, in below case, default value = 0
                            fy.setdefault(feature, 0)
                            fy[feature] += 1

                        # Calculate the features for the next best tag (z_k)
                        f = self.feature_vector(dict(t2=z[k], t1=z[k+1], x=x, i=k+1), z[k+2])

                        # Add the features to the best tagging feature vector
                        for feature in f:
                            fz.setdefault(feature, 0)
                            fz[feature] += 1

                    # Update the feature vector (adding new features if necessary)
                    for feature in fy:
                        self.v.setdefault(feature, 0)
                        self.v[feature] += fy[feature]
                    for feature in fz:
                        self.v.setdefault(feature, 0)
                        self.v[feature] -= fz[feature]

            if debug:
                #sys.stdout.write("Updated weight vector:\n")
                self.write_weight_vector()

def feature_vector(history, tag):
        """
        Compute the feature vector g(h, t) for the given history/tag pair.
        """
        # Store the feature vector as a map from feature strings to counts.
        g = {}

        # Generate the possible features to search
        f = set()

        # Trigram feature
        f.add(TRIGRAM+':'+history['t2']+':'+history['t1']+':'+tag)

        # Tag feature (if not beyond the end of the sentence)
        if history['i'] <= len(history['x']):
            # history['x'][history['i'] = ['i','am','wasifa'][1-1] = ['i','am','wasifa'][0] 
            # so we are getting that word (item in a list) from the x list (each sentence) using the index, [history['i'] - 1]
            # so index [1 - 1] = index 0 would return word 'i' from the list sentence x.
            f.add(TAG+':'+history['x'][history['i'] - 1]+':'+tag)

        # Suffix features (if not beyond the end of the sentence)
        #if history['i'] <= len(history['x']):
        #    # for j=1:3-1
        #    for j in range(1,3):
        #        word = history['x'][history['i'] - 1]
        #        if j <= len(word):
        #            # word[-j:] where -j when j=1 refers to the last element 
        #            f.add(SUFFIX+':'+word[-j:]+':'+str(j)+':'+tag)

        # Check each feature
        for feature in f:
            token = feature.split(':')

            # Calculate trigram features
            if token[0] == TRIGRAM:
                s = token[1]
                u = token[2]
                v = token[3]
                if history['t2'] == s and history['t1'] == u and tag == v:
                    # g is a dictionary storing the number of times that each feature occured (value) using as index (key) the feature string, e.g 'TRIGRAM:'*':'*':0'
                    g[feature] = 1

            # Calculate tag features
            if token[0] == TAG:
                # Handle cases where ':' appears as part of the word
                # join() method returns a string by joining a sequence of elements using a separator
                # ("-").join(("a", "b", "c")) would return string a-b-c
                r = ':'.join(token[1:-1])
                u = token[-1]
                if history['i'] <= len(history['x']):
                    w = history['x'][history['i'] - 1]
                else:
                    w = ''
                if w == r and tag == u:
                    g[feature] = 1


            ## Calculate suffix features
            #if token[0] == SUFFIX:
            #    # Handle cases where ':' appears as part of the word
            #    u = ':'.join(token[1:-2])
            #    j = int(token[-2])
            #    v = token[-1]
            #    if history['i'] <= len(history['x']):
            #        w = history['x'][history['i'] - 1]
            #    else:
            #        w = ''
            #    if w[-j:] == u and tag == v:
            #        g[feature] = 1

        return g

def inner_product(v, g):
        return sum((v.get(key, 0) * value for key, value in g.iteritems()))

def viterbi_algorithm(tags, x, fv):
    """
    Run the GLM form of the Viterbi algorithm on the input sentence (x)
    using the previously determined feature (f) and weight (v) vectors.
    """
    #total number of words in sentence x
    n = len(x)

    # Initialise pi and bp
    pi = {}
    bp = {}
    # pi{key:value} where key is a tuple, that is, (k th position, tags in k-1 position, tags in kth position)
    # so, pi[(0, '*', '*')] means pi[key]=pi[(k,T,U)] where k=0,T='*',U='*'
    pi[(0, '*', '*')] = 0

    # for k=1:n+1-1 in MATLAB style
    # that is, for each word, k, in the sentence
    for k in range(1, n+1):

        if k == 1:
            # T is possible tags of second previous word in position k-2
            T = {'*'}
            # U is possible tags of first previous word in position k-1
            U = {'*'}
            # S is possible tags of the current word in position k
            S = tags
        elif k == 2:
            T = {'*'}
            U = tags
            S = tags
        else:
            T = tags
            U = tags
            S = tags

        for u in U:
            for s in S:
                pi[(k, u, s)], bp[(k, u, s)] = max(((pi[(k-1, t, u)] + inner_product(fv,feature_vector(dict(t2=t, t1=u, x=x, i=k), s)), t) for t in T))

    # Store the tag sequence as an array
    tag = ['']*(n+1)

    # Calculate the tag sequence by following the back pointers
    prob, tag[n-1], tag[n] = max((max(((pi[(n, u, s)] + inner_product(fv,feature_vector(dict(t2=u, t1=s, x=x, i=n+1), STOP)), u, s) for s in S)) for u in U))
    for k in range(n-2, 0, -1):
        tag[k] = bp[(k+2, tag[k+1], tag[k+2])]
    tag = tag[1:]

    # Return the probability and tag sequence
    return prob, tag

def write_weight_vector(v):
    """
    """
    fo = open("/Users/vivian/NER/data/test.out", "wb")
    for feature in v:
        fo.write("%s %.1f\n" % (feature, v[feature]))
    fo.close()

def read_training_data(training_file):
        """
        """
        x = []
        y = []
        tagset = set()
        if debug: sys.stdout.write("Reading training data...\n")

        file = open(training_file, 'r')
        # file.seek(0) sets the file's current reading position at offset 0
        file.seek(0)
        sentence = []
        tags = []
        for line in file:
            #String.strip([chars]);
            #returns a copy of the string in which all chars have been stripped from the beginning and the end of the string.
            #e.g str = "0000000this is string example....wow!!!0000000"; 
            #    print str.strip( '0' )
            # answer: this is string example....wow!!!
            line = line.strip()
            if line: # Non-empty line
                #method split() returns a list of all the words in the string, using a delimiter (splits on all whitespace if left unspecified)
                token = line.split()
                word = token[0]
                tag  = token[3]
                #appends the word at the end of the list
                sentence.append(word)
                #appends the tag at the end of the list
                tags.append(tag)
            else: # End of sentence reached
                #converts the sentence list to a tuple and then appends that tuple to the end of the self.x list
                x.append(tuple(sentence))
                y.append(tuple(tags))
                #set object is an unordered collection of items and don't have duplicate items
                #set.update(other, ...)
                #Update the set, adding elements from all others
                # so self.tags.update(tags) keeps only the distinct # of tags for that sentence, that is, set([0,I-Gene]) for all sentences 
                tagset.update(tags)
                sentence = []
                tags = []
        file.close()
        return x,y,tagset

def perc_train(train_data, tags, iterations):
    fv = defaultdict(int)
    # range(iterations) = [0,1,2,3,4]
    # for iteration=0:4 in MATLAB style
    #t0 = time.time()
    for iteration in range(5):
        t0=time.time()
        #if debug: sys.stdout.write("Perceptron algorithm iteration %d...\n" % (iteration+1))

        # Loop over each sentence/tag pair (xi, yi) in the training data

        # len(self.x) returns total number of sentences in training data
        # range(len(self.x)) = [0,1,2...,total number of sentences in training data-1]
        # for i=0:total number of sentences in training data-1 in MATLAB style
        for (a,b) in train_data:
            x = list(a)
            y = list(b)

            # Find the best tagging sequence using the Viterbi algorithm
            prob, z = viterbi_algorithm(tags, x, fv)

            # Check if the tags match the gold standard
            if z != y:

                # Compute the gold tagging feature vector f(x, y)
                # and the best tagging feature vector f(x, z)
                fy = {}
                fz = {}

                # Add the start and stop symbols to the tagging sequences
                x = x + [STOP]
                y = [START, START] + y + [STOP]
                z = [START, START] + z + [STOP]

                # Loop over all words in the sentence
                for k in range(len(x)):

                    # Calculate the features for the next gold tag (y_k)
                    f = feature_vector(dict(t2=y[k], t1=y[k+1], x=x, i=k+1), y[k+2])

                    # Add the features to the gold tagging feature vector
                    for feature in f:
                        # This method returns the key value available in the dictionary and if given key is not available then it will return provided default value, in below case, default value = 0
                        fy.setdefault(feature, 0)
                        fy[feature] += 1

                    # Calculate the features for the next best tag (z_k)
                    f = feature_vector(dict(t2=z[k], t1=z[k+1], x=x, i=k+1), z[k+2])

                    # Add the features to the best tagging feature vector
                    for feature in f:
                        fz.setdefault(feature, 0)
                        fz[feature] += 1

                # Update the feature vector (adding new features if necessary)
                for feature in fy:
                    fv[feature] += fy[feature]
                for feature in fz:
                    fv[feature] -= fz[feature]
        print ("iteration: %d with %f seconds" % (iteration,time.time() - t0))
    return fv.items()
def main(training_file):

    iterations = 5;
    x,y,tags = read_training_data(training_file)
    v = {}
    sc = SparkContext(appName="parameterMixing")
    tags = sc.broadcast(tags)
    training_data = []
    for i in range(len(x)):
        training_data.append((x[i],y[i]))
    train_data = sc.parallelize(training_data).cache()
    feat_vec_list = train_data.mapPartitions(lambda t: perc_train(t, tags.value, iterations))
    feat_vec_list = feat_vec_list.combineByKey((lambda x: (x,1)),
                             (lambda x, y: (x[0] + y, x[1] + 1)),
                             (lambda x, y: (x[0] + y[0], x[1] + y[1]))).collect()

    sc.stop()
    for (feat, (a,b)) in feat_vec_list:
        v[feat] = float(a)/float(b)
    # Compute the weight vector using the Perceptron algorithm
    #trainer.perceptron_algorithm(5)

    # Write out the final weight vector
    write_weight_vector(v)

def usage():
    sys.stderr.write("""
    Usage: python perceptron_training.py [training_file]
        Find the most likely tag sequence for each sentence in the input
        data file using a Global Linear Model decoder with a previously
        determined weight vector.\n""")

if __name__ == "__main__":
    main('/Users/vivian/NER/data/eng_processed.train.txt')
    #main("eng.train")