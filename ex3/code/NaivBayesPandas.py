import pandas as pd
import os
from typing import List
import re
import glob
import numpy as np
from collections import Counter

def extractWords(filecontent: str) -> List[str]:
    '''
    Word extractor from filecontent
    :param filecontent: filecontent as a string
    :return: list of words found in the file
    '''
    txt = filecontent.split(" ")
    txtClean = [(re.sub(r'[^a-zA-Z]+', '', i).lower()) for i in txt]
    words = [i for i in txtClean if i.isalpha()]
    return words

def importFiles(filename: str):
    spam_words = []
    ham_words = []
    files = sorted(glob.glob(filename + '*.txt'))
    for f in files:
        # open training files and save content as String
        with open(f, 'r') as myfile:
            emailContent = myfile.read()

        # extract every word in words list
        words = extractWords(emailContent)

        if 'spmsga' in f:
            spam_words.extend(words)
        else:
            ham_words.extend(words)

    return (spam_words, ham_words)



filedir = '../data/emails/'
spam, ham = importFiles(os.path.join(filedir, 'train/'))

spam_words = Counter(spam)
ham_words = Counter(ham)

#df = pd.DataFrame([spam_words], columns = ['word', 'ham', 'spam'])
df = pd.DataFrame.from_dict(spam_words)



print(df)