import nltk
import numpy as np
from nltk.corpus import genesis
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('genesis')

genesis_ic = wn.ic(genesis, False, 0.0)

def wup(S1, S2):
    """Wu-Palmer similarity."""
    return S1.wup_similarity(S2)

def resnik(S1, S2):
    """Resnik similarity."""
    return S1.res_similarity(S2, genesis_ic)

options = {0: wup, 1: resnik}

def preProcess(sentence):
    """Tokenize, remove stopwords, and clean the sentence."""
    Stopwords = list(set(nltk.corpus.stopwords.words('english')))
    words = word_tokenize(sentence)
    words = [word.lower() for word in words if word.isalpha() and word not in Stopwords] 
    return words

def get_wordnet_pos(word):
    """Map POS tag to first character for lemmatization with WordNet."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wn.ADJ, "N": wn.NOUN, "V": wn.VERB, "R": wn.ADV}
    return tag_dict.get(tag, wn.NOUN)  

def word_similarity(w1, w2, num):
    """Calculate similarity between two words only if they share the same POS."""
    pos1 = get_wordnet_pos(w1)
    pos2 = get_wordnet_pos(w2)

    synsets1 = wn.synsets(w1, pos=pos1)
    synsets2 = wn.synsets(w2, pos=pos2)
    
    if synsets1 and synsets2:
        S1 = synsets1[0]  
        S2 = synsets2[0]  
        try:
            similarity = options[num](S1, S2)
            if similarity:
                return round(similarity, 2)
        except nltk.corpus.reader.wordnet.WordNetError:
            return 0
    return 0

def Similarity(T1, T2, num):
    """Calculate sentence-to-sentence similarity using TF-IDF and WordNet similarity."""
    words1 = preProcess(T1)
    words2 = preProcess(T2)

    tf = TfidfVectorizer(use_idf=True)
    tf.fit_transform([' '.join(words1), ' '.join(words2)])
    
    Idf = dict(zip(tf.get_feature_names_out(), tf.idf_))
    
    Sim_score1 = 0
    Sim_score2 = 0

    for w1 in words1:
        Max = 0
        for w2 in words2:
            score = word_similarity(w1, w2, num)
            if Max < score:
                Max = score
        Sim_score1 += Max * Idf.get(w1, 0)
    Sim_score1 /= sum([Idf.get(w1, 0) for w1 in words1])

    for w2 in words2:
        Max = 0
        for w1 in words1:
            score = word_similarity(w1, w2, num)
            if Max < score:
                Max = score
        Sim_score2 += Max * Idf.get(w2, 0)
    Sim_score2 /= sum([Idf.get(w2, 0) for w2 in words2])

    Sim = (Sim_score1 + Sim_score2) / 2
    
    return round(Sim, 2)

if __name__=='__main__':
    T1 = 'Students feel unhappy today about the class of 2020'
    T2 = 'Many students struggled to understand some key concepts about the subject seen in the class of 2020'
    print('Wup Similarity(T1, T2) =', Similarity(T1, T2, 0))
    print('Resnik Similarity(T1, T2) =', Similarity(T1, T2, 1))
