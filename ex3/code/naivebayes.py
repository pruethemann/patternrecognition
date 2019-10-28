import numpy as np
import math
import glob
import re
from typing import List



class wordCounter():
    '''
    Placeholder to store information about each entry (word) in a dictionary
    '''

    def __init__(self, word, numOfHamWords, numOfSpamWords, p):
        self.word = word
        self.numOfHamWords = numOfHamWords
        self.numOfSpamWords = numOfSpamWords
        self.p = p


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
        ham_words = {}
        spam_words = {}
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
                if 'spmsga' in f: 
                    if w not in spam_words:
                        spam_words[w] = 1
                    
                    else:
                        spam_words[w] += 1
                    
                else:
                    if w not in ham_words:
                        ham_words[w] = 1
                    
                    else:
                        ham_words[w] += 1
        
        final_dictionary = []
        self.totalNumberSpamWords = 0
        self.totalNumberHamWords = 0
        for spam in spam_words:
            numberOfSpamWords = spam_words[spam]
            self.totalNumberSpamWords += numberOfSpamWords
            final_dictionary.append(wordCounter(spam, 0, numberOfSpamWords, 0))
        
        for ham in ham_words:            
            numberOfHamWords = ham_words[ham]
            self.totalNumberHamWords += numberOfHamWords
            # create a new wordCounter instance for every new word
            if not self.checkInDictionary(final_dictionary, ham):
                newWord = wordCounter(ham, numberOfHamWords, 0, 0)
                final_dictionary.append(newWord)
                
            else:
                existingWord = self.getWordCounter(final_dictionary, ham, len(final_dictionary), True)
                existingWord.numOfHamWords = numberOfHamWords 
                final_dictionary.append(existingWord)
                
        ## calculate probability of being spam per word
        for e in final_dictionary:
            e.p = e.numOfSpamWords / (e.numOfSpamWords + e.numOfHamWords)
        
        
        priorSpam =   spam_count/ len(files)   
        
        self.spam_count = spam_count
        self.ham_count = len(files) - spam_count

        self.logPrior = math.log(priorSpam / (1.0 - priorSpam))
        
        self.logSpamPrior = np.log(priorSpam)
        self.logHamPrior = np.log(1- priorSpam)        
        
        final_dictionary.sort(key=lambda x: x.p, reverse=True)
        self.dictionary = final_dictionary
        return self.dictionary, self.logPrior
    
    def checkInDictionary(self, final_dict, word) -> bool:

        for i in range(len(final_dict)):
            if final_dict[i].word == word:
                return True

        return False
            
    def getWordCounter(self, final_dict, word, number_of_features, remove = False) -> wordCounter:
        c = 0
        for i in range(number_of_features):
            c+=1
            if final_dict[i].word == word:
                element = final_dict[i]
                if(remove):
                    final_dict.remove(element)
              #  print(c, "   ", number_of_features)
                return element
            
        for i in range(len(final_dict)- number_of_features, len(final_dict)):
            if final_dict[i].word == word:
                element = final_dict[i]
                if(remove):
                    final_dict.remove(element)
                return element            

        return None
        

    def classify(self, message: str, number_of_features: int) -> bool:
        '''
        :param message: Input email message as a string
        :param number_of_features: Number of features to be used from the trained dictionary
        :return: True if classified as SPAM and False if classified as HAM
        '''
        txt = np.array(self._extractWords(message))
        
        # TODO: Implement classification function
        spamLogL = self.logSpamPrior
        hamLogL = self.logHamPrior
        
        for word in txt:
            wordStats = self.getWordCounter(self.dictionary, word, number_of_features)
            if wordStats == None:
                spamLogL += np.log(1/self.totalNumberSpamWords+1)
                hamLogL += np.log(1/self.totalNumberHamWords    +1)
            else:
#                print(word, "   :", (wordStats.numOfSpamWords), "  ",  (self.totalNumberSpamWords) )
#                print(word, "   :", (wordStats.numOfHamWords), "  ",  (self.totalNumberHamWords) )                
                spamLogL += np.log((wordStats.numOfSpamWords+1) / (self.totalNumberSpamWords+1))
                hamLogL += np.log((wordStats.numOfHamWords+1) / (self.totalNumberHamWords    +1)        )
        
        #print(spamLogL, "   ", hamLogL)
        if spamLogL > hamLogL:
            return True
        return False

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
              
             ## check if email is spamm
            spam = self.classify(emailContent, number_of_features)
            if spam and 'spmsga' in f or not spam and 'spmsga' not in f:

                corr += 1 
                            
            else:
               #♥ print(f, "   ", spam)
                ncorr +=1
        return corr / (corr + ncorr)
    

    def printMostPopularSpamWords(self, num: int) -> None:
        print("{} most popular SPAM words:".format(num))
        # print the 'num' most used SPAM words from the dictionary       
        self.dictionary.sort(key=lambda x: x.numOfSpamWords, reverse=True)
        
        c = 0
        for spam in self.dictionary:
            if c > num:
                break
            print(spam.word, "   ", spam.numOfHamWords, "   ", spam.numOfSpamWords, "   " , round(spam.p,2))
            c +=1

    def printMostPopularHamWords(self, num: int) -> None:
        print("{} most popular HAM words:".format(num))
        # print the 'num' most used HAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.numOfHamWords, reverse=True)
        
        c = 0
        for ham in self.dictionary:
            if c > num:
                break
            print(ham.word, "   ", ham.numOfHamWords, "   ", ham.numOfSpamWords, "   " , round(ham.p,2))
            c +=1        
        

    def printMostindicativeSpamWords(self, num: int) -> None:
        print("{} most distinct SPAM words:".format(num))
        # TODO: print the 'num' most indicative SPAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.p, reverse=True)
        
        c = 0
        for spam in self.dictionary:
            if c > num:
                break
            print(spam.word, "   ", spam.numOfHamWords, "   ", spam.numOfSpamWords, "   " , round(spam.p,2))
            c +=1         
        

    def printMostindicativeHamWords(self, num: int) -> None:
        print("{} most distinct HAM words:".format(num))
        # TODO: print the 'num' most indicative HAM words from the dictionary
        self.dictionary.sort(key=lambda x: x.p, reverse=False)
        
        c = 0
        for ham in self.dictionary:
            if c > num:
                break
            print(ham.word, "   ", ham.numOfHamWords, "   ", ham.numOfSpamWords, "   " , round(ham.p,2))
            c +=1   