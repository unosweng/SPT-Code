"""
Hugging Face BERT with custom classifier (PyTorch)

https://www.kaggle.com/code/angyalfold/hugging-face-bert-with-custom-classifier-pytorch/notebook
"""

import re
import string


def remove_urls(tweet):
    return re.sub(r"http(s?)://[\S]+", '', tweet)

def remove_at_links(tweet):
    return re.sub(r"\B(@)\S+", '', tweet)

def remove_non_ascii_chars(tweet):
    ascii_chars = set(string.printable)
    for c in tweet:
        if c not in ascii_chars:
            tweet = tweet.replace(c,'')
    return tweet

def fix_ax_nots(tweet):
    tweet = tweet.replace(" dont ", " do not ")
    tweet = tweet.replace(" don't ", " do not ")
    tweet = tweet.replace(" doesnt ", " does not ")
    tweet = tweet.replace(" doesn't ", " does not ")
    tweet = tweet.replace(" wont ", " will not ")
    tweet = tweet.replace(" won't ", " will not ")
    tweet = tweet.replace(" cant ", " cannot ")
    tweet = tweet.replace(" can't ", " cannot ")
    tweet = tweet.replace(" couldnt ", " could not ")
    tweet = tweet.replace(" couldn't ", " could not ")
    tweet = tweet.replace(" shouldnt ", " should not ")
    tweet = tweet.replace(" shouldn't ", " should not ")
    tweet = tweet.replace(" wouldnt ", " would not ")
    tweet = tweet.replace(" wouldn't ", " would not ")
    tweet = tweet.replace(" mustnt ", " must not ")
    tweet = tweet.replace(" mustn't ", " must not ")
    return tweet

def fix_personal_pronouns_and_verb(tweet):
    tweet = tweet.replace(" im ", " i am ")
    tweet = tweet.replace(" youre ", " you are")
    tweet = tweet.replace(" hes ", " he is") # ? he's can be he has as well
    tweet = tweet.replace(" shes ", " she is")
    # we are -> we're -> were  ---- were is a valid word
    tweet = tweet.replace(" theyre ", " they are")
    
    tweet = tweet.replace(" ive ", " i have ")
    tweet = tweet.replace(" youve ", " you have ")
    tweet = tweet.replace(" weve ", " we have ")
    tweet = tweet.replace(" theyve ", " they have ")
    
    tweet = tweet.replace(" youll ", " you will ")
    tweet = tweet.replace(" theyll ", " they will ")
    
    return tweet

def fix_special_chars(tweet):
    tweet = tweet.replace("&amp;", " and ")
    # tweet = tweet.replace("--&gt;", "")
    return tweet
        

def clean_tweet(tweet):
    tweet = remove_urls(tweet)
    tweet = remove_at_links(tweet)
    tweet = remove_non_ascii_chars(tweet)
    tweet = fix_special_chars(tweet)
    tweet = fix_ax_nots(tweet)
    tweet = fix_personal_pronouns_and_verb(tweet)
    return tweet