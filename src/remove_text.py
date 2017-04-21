import string

from nltk.corpus import stopwords
from nltk.stem.snowball import PorterStemmer
from nltk.tokenize.casual import TweetTokenizer
import nlp_utils

stemmer = PorterStemmer()
tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True)

tweet=''

tokens = tokenizer.tokenize(tweet)
text_proc = []
for token in tokens:
    token = token.strip()
    if len(token) < 3:
        continue
    elif token in stopwords.words('english'):
        continue
    elif nlp_utils.match_url(token):
        continue
    elif token in string.punctuation:
        continue
    # elif token.startswith(("#", "$")):
    #     continue

    token = token.translate({ord(k): "" for k in string.punctuation})
    token = stemmer.stem(token)

    token = token.strip()
    if token == "":
        continue

    text_proc.append(token)