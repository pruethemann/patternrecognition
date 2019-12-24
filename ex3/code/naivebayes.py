import numpy as np
import math
import glob
import re
from typing import List



class wordCounter():
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''

    def __init__(self, word, numOfHamWords, numOfSpamWords, p=0):
        self.word = word
        self.numOfHamWords = numOfHamWords
        self.numOfSpamWords = numOfSpamWords
        self.p = p ## probability of being SPAM: numSpam / (numSpam + numHam)


class naiveBayes():
    '''
    Naive bayes class
    Train model and classify new emails
    '''

    def _extractWords(self, filecontent: str) -> List[str]:
        '''
        Word extractor from filecontent
        :param filecontent: filecontent as a string
        :return: list of words found in the file
        '''
        txt = filecontent.split(" ")
        txtClean = [(re.sub(r'[^a-zA-Z]+', '', i).lower()) for i in txt]
        words = [i for i in txtClean if i.isalpha()]
        return words

    def train(self, msgDirectory: str, fileFormat: str = '*.txt') -> (List[wordCounter], float):
        '''
        :param msgDirectory: Directory to email files that should be used to train the model
        :return: model dictionary and model prior
        '''
        files = sorted(glob.glob(msgDirectory + fileFormat))

        # TODO: Train the naive bayes classifier
        # TODO: Hint - store the dictionary as a list of 'wordCounter' objects
        word_counter = {}

        spam_count = 0

        for f in files:
            # open training files and save content as String
            with open(f, 'r') as myfile:
                emailContent = myfile.read()

            # extract every word in words list
            words = self._extractWords(emailContent)
            if 'spmsga' in f:
                spam_count += 1

            ## Classify every word into ham or spam based on training set
            for w in words:
                ## Training Spam emails start with  's' in filename
                if w not in word_counter:
                    if 'spmsga' in f:
                        word_counter[w] = wordCounter(w, 1, 2)
                    else:
                        word_counter[w] = wordCounter(w, 2, 1)
                else:
                    if 'spmsga' in f:
                        word_counter[w].numOfSpamWords += 1
                    else:
                        word_counter[w].numOfHamWords += 1

        ## calculate probability of being spam per word
        for word in word_counter:
            word_counter[word].p = word_counter[word].numOfSpamWords / (word_counter[word].numOfSpamWords + word_counter[word].numOfHamWords)

        priorSpam = spam_count / len(files)

        self.spam_count = spam_count
        self.ham_count = len(files) - spam_count

        ## prior: priorSpam / priorHam
        self.logPrior = math.log(priorSpam / (1.0 - priorSpam))
        #self.logSpamPrior = math.log(priorSpam)
        #self.logHamPrior = np.log(1 - priorSpam)


        words_list = [word_counter[w] for w in word_counter]

        words_list.sort(key=lambda x: x.p, reverse=True)
        self.dictionary = words_list

    def getSpamScore(self, word, number_of_features):

        ## check first num of indicative spam words
        for i in range(number_of_features):
            if self.dictionary[i].word == word:
                ## return ratio of Spam/Ham REMEMBER: normalisation term (numSpam + numHam) cancels
                return (self.dictionary[i].numOfSpamWords) / (self.dictionary[i].numOfHamWords)

        ## check first num of indicative ham words
        for i in range(len(self.dictionary) - number_of_features - 1, len(self.dictionary)):
            if self.dictionary[i].word == word:
                return (self.dictionary[i].numOfSpamWords) / (self.dictionary[i].numOfHamWords)

        ## if word is not in dicionary. Return 1 -> log(1) = 0 Laplace smoothing
        return 1

    def classify(self, message: str, number_of_features: int) -> bool:
        '''
        :param message: Input email message as a string
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''
        ## Import all words of specific email
        words = np.array(self._extractWords(message))

        # Implement classification function: Naive Bayes S. 12
        score = self.logPrior ## log( prior(Spam) / prior(Ham))

        for w in words:
            ## The higher the score, the more likely it's spam
            wordScore = self.getSpamScore(w, number_of_features)
            score += np.log(wordScore)

        ## if larger 0. Spam wins
        return score > 0


    def classifyAndEvaluateAllInFolder(self, msgDirectory: str, number_of_features: int,
                                       fileFormat: str = '*.txt') -> float:
        '''
        :param msgDirectory: Directory to email files that should be classified
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: Classification accuracy
        '''
        
        files = sorted(glob.glob(msgDirectory + fileFormat))
        corr = 0  # Number of correctly classified messages
        ncorr = 0  # Number of falsely classified messages
        # TODO: Classify each email found in the given directory and figure out if they are correctly or falsely classified
        # TODO: Hint - look at the filenames to figure out the ground truth label
        for f in files:
            # open training files and save content as String
            with open(f, 'r') as myfile:
              emailContent = myfile.read()
              
             ## check if email is spam
            is_spam = self.classify(emailContent, number_of_features)
            if is_spam and 'spmsga' in f or not is_spam and 'spmsga' not in f:
                corr += 1
            else:
                ncorr +=1
        return corr / (corr + ncorr)
    

    def printMostPopularSpamWords(self, num: int) -> None:
        print("\n{} most popular SPAM words:".format(num))
        # print the 'num' most used SPAM words from the dictionary
        ## sort for largest number of Spam
        self.dictionary.sort(key=lambda x: x.numOfSpamWords, reverse=True)
        self.printWords(num)

    def printMostPopularHamWords(self, num: int) -> None:
        print("\n{} most popular HAM words:".format(num))
        # print the 'num' most used HAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.numOfHamWords, reverse=True)
        self.printWords(num)

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("\n{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.p, reverse=True)
        self.printWords(num)
        

    def printMostindicativeHamWords(self, num: int) -> None:
        print("\n{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.p, reverse=False)
        self.printWords(num)

    def printWords(self, num:int) -> None:
        for i in range(num):
            w = self.dictionary[i]
            print(f'{w.word} Ham: {w.numOfHamWords} Spam: {w.numOfSpamWords} Ind: {round(w.p,2)}')