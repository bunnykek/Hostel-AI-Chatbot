# things we need for NLP
import os
import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

import json

from mongoDB import mongodb

ERROR_THRESHOLD = 0.25
CONFIDENCE_THRESHOLD = 0.5


class Chatbot:
    def __init__(self, retrain: bool = False) -> None:
        with open('intents.json') as json_data:
            self.intents = json.load(json_data)
        self.words = []
        self.classes = []
        self.documents = []
        self.ignore_words = ['?']#, 'what', 'which', 'who', 'where', 'why', 'when', 'how', 'whose', 'i', 'me', 'they', 'you', 'a', 'there', 'there', 'are', 'on', 'in', 'the', 'an', 'of', 'or', 'and']
        # loop through each sentence in our intents patterns
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                # tokenize each word in the sentence
                w = nltk.word_tokenize(pattern)
                # add to our words list
                self.words.extend(w)
                # add to documents in our corpus
                self.documents.append((w, intent['tag']))
                # add to our classes list
                if intent['tag'] not in self.classes:
                    self.classes.append(intent['tag'])

        # stem and lower each word and remove duplicates
        self.words = [stemmer.stem(w.lower()) for w in self.words if w not in self.ignore_words]
        self.words = sorted(list(set(self.words)))

        # remove duplicates
        self.classes = sorted(list(set(self.classes)))

        print (len(self.documents), "documents")
        print (len(self.classes), "classes", self.classes)
        print (len(self.words), "unique stemmed words", self.words)

        # create our training data
        training = []
        output = []
        # create an empty array for our output
        output_empty = [0] * len(self.classes)

        # training set, bag of words for each sentence
        for doc in self.documents:
            # initialize our bag of words
            bag = []
            # list of tokenized words for the pattern
            pattern_words = doc[0]
            # stem each word
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            # create our bag of words array
            for w in self.words:
                bag.append(1) if w in pattern_words else bag.append(0)

            # output is a '0' for each tag and '1' for current tag
            output_row = list(output_empty)
            output_row[self.classes.index(doc[1])] = 1

            training.append([bag, output_row])

        # shuffle our features and turn into np.array
        random.shuffle(training)
        training = np.array(training)

        # create train and test lists
        train_x = list(training[:,0])
        train_y = list(training[:,1])

        tf.compat.v1.reset_default_graph()

        # reset underlying graph data
        # tf.reset_default_graph()
        # Build neural network
        net = tflearn.input_data(shape=[None, len(train_x[0])])
        net = tflearn.fully_connected(net, 128)
        net = tflearn.fully_connected(net, 128)
        net = tflearn.fully_connected(net, 128)
        # net = tflearn.fully_connected(net, 128)
        # net = tflearn.fully_connected(net, 128)
        # net = tflearn.fully_connected(net, 128)
        # net = tflearn.fully_connected(net, 128)
        # net = tflearn.fully_connected(net, 128)
        net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
        net = tflearn.regression(net)

        # Define model and setup tensorboard
        self.model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

        if retrain:
            tf.compat.v1.reset_default_graph()
            self.model.fit(train_x, train_y, n_epoch=1000, batch_size=128, show_metric=True)
            self.model.save(os.path.join("trainedData", "model.tflearn"))
            return self.model
        
        try:
            self.model.load(os.path.join("trainedData", "model.tflearn"))
        except Exception as e:
            print("Some error occurred!.. Retraining the model")
            self.model = self.__init__(retrain=True)

        
    
    # def retrain(self):
    #     self.__init__()
    #     # Start training (apply gradient descent algorithm)
        

    def __clean_up_sentence(self, sentence):
        # tokenize the pattern
        sentence_words = nltk.word_tokenize(sentence)
        # stem each word
        sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
        return sentence_words

    # return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
    def __bow(self, sentence, words, show_details=False):
        # tokenize the pattern
        sentence_words = self.__clean_up_sentence(sentence)
        # bag of words
        bag = [0]*len(words)  
        for s in sentence_words:
            for i,w in enumerate(words):
                if w == s: 
                    bag[i] = 1
                    if show_details:
                        print ("found in bag: %s" % w)

        return(np.array(bag))


    def __classify(self, sentence):
        # generate probabilities from the model
        results = self.model.predict([self.__bow(sentence, self.words)])[0]
        # print(results)
        # filter out predictions below a threshold
        results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        # return tuple of intent and probability
        return return_list

    def response(self, sentence):
        results = self.__classify(sentence)
        print(results)
        if not len(results) or results[0][1] < CONFIDENCE_THRESHOLD:
            print("New kind of query!!")
            if not mongodb.count_documents({"query": sentence}):
                mongodb.insert_one({"query": sentence})
            return None
        # if we have a classification then find the matching intent tag
        if results:
            # loop as long as there are matches to process
            while results:
                for i in self.intents['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # a random response from the intent
                        return random.choice(i['responses'])

                results.pop(0)


if __name__ == "__main__":
    bot = Chatbot()
    # bot.retrain()
    # print(bot.__classify("Where are the water filters?"))
    print(bot.response("Where are the water filters?"))
