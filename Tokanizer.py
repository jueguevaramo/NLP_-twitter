import nltk
from nltk.tokenize import TweetTokenizer


class TOC():
    """
    Types_tokenizer
    """

    def __init__(self, seq):
        super(TOC, self).__init__()
        self.seq = seq

    def from_regex(self):
        pattern = r'''(?x)          # set flag to allow verbose regexps
                (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
              | \w+(?:-\w+)*        # words with optional internal hyphens
              | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82
              | \.\.\.              # ellipsis
              | [][.,;"'?():_`-]'''

        return(nltk.regexp_tokenize(self.seq, pattern))

    def from_tweet(self):
        tk = TweetTokenizer()
        return(tk.tokenize(self.seq))
