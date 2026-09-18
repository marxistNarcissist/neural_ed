import joblib, tiktoken
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ENC = tiktoken.get_encoding("cl100k_base")


def encode_corpus(docs):
    """['I love NLP'] -> ['40 3021 452 12852']  (IDs as space-joined string)"""
    return [" ".join(map(str, ENC.encode(d))) for d in docs]


class NlpEncoder:
    def __init__(self, ngram_range=(1, 2), min_df=1, max_features=None):
        self.vec = TfidfVectorizer(
            token_pattern=r"\S+", 
            lowercase=False,  
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=True,
        )

    def fit(self, docs):
        self.vec.fit(encode_corpus(docs))
        return self

    def transform(self, docs):
        return self.vec.transform(encode_corpus(docs))   # sparse matrix

    def nlp_encode(self, docs):
        np.set_printoptions(
            precision=3,
            suppress=True,
            linewidth=np.inf,
            threshold=np.inf,
        )
        return self.vec.fit_transform(encode_corpus(docs)).toarray()


    # ---- inspection helpers ----
    def feature_names(self):
        return self.vec.get_feature_names_out()

    def readable(self, feature):
        return "".join(ENC.decode([int(t)]) for t in feature.split())

    def top_terms(self, doc, k=5):
        
        row = self.transform([doc]).toarray()[0]
        names = self.feature_names()
        idx = row.argsort()[-k:][::-1]
        return [(self.readable(names[i]), round(row[i], 3))
                for i in idx if row[i] > 0]

    def save(self, path):  joblib.dump(self.vec, path)
    def load(self, path):  self.vec = joblib.load(path); return self