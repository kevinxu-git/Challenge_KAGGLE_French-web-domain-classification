import re 
import nltk
import string
from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')

stpwds = stopwords.words('french')
punct = string.punctuation
punct = string.punctuation.replace('-', '')

def clean_text_website_simple(text, my_stopwords, punct, remove_stopwords=True, stemming=True):
    text_website = text.lower()
    text_website = re.sub(' +',' ',text_website) # strip extra white space
    text_website = text_website.strip() # strip leading and trailing white space
    text_website = ''.join(l for l in text_website if l not in punct) # remove punctuation (preserving intra-word dashes)
    tokens = text_website.split(' ') # tokenize (split based on whitespace)
    if remove_stopwords:
        tokens = [token for token in tokens if token not in my_stopwords]
    if stemming:
        stemmer = nltk.stem.SnowballStemmer("french")
        tokens_stemmed = list()
        for token in tokens:
            tokens_stemmed.append(stemmer.stem(token))
        tokens = tokens_stemmed
    return tokens
