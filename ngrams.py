"""Conservative repair of the coursework's counting + maximum-probability backoff."""
import random
import re
import html
import unicodedata
from collections import Counter

def preprocess(text):
    """2026 lexical cleaner: normalize words and mask URL/mention tokens.

    This scans semantic token types once rather than applying the course's
    ordered substitution table. Punctuation and numbers separate words.
    """
    if not isinstance(text,str): raise TypeError('text must be a string')
    normalized=unicodedata.normalize('NFKC',html.unescape(text)).casefold()
    tokens=re.findall(r"https?://[^\s]+|www\.[^\s]+|@\w+|[^\W\d_]+(?:['’][^\W\d_]+)*",normalized)
    words=[]
    for token in tokens:
        if not words and token=='rt': continue
        if token.startswith(('http://','https://','www.')): words.append('link')
        elif token.startswith('@'): words.append('user')
        else: words.append(token.replace('’',"'"))
    return ' '.join(words)

def tokenize(text):
    from nltk.tokenize import word_tokenize
    # Same Treebank tokenizer; no external Punkt download for sentence splitting.
    return word_tokenize(preprocess(text),preserve_line=True)

def make_grams(tokens,n=2):
    if n not in (1,2,3): raise ValueError('n must be 1, 2 or 3')
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def train(sentences):
    rows=[tokenize(s) for s in sentences]
    if not rows: raise ValueError('training corpus is empty')
    if any(t in ('<s>','</s>','<unk>') for row in rows for t in row):
        raise ValueError('reserved boundary tokens are not corpus words')
    vocabulary={t:i for i,t in enumerate(['<s>','</s>','<unk>']+sorted({t for row in rows for t in row}))}
    counts={n:Counter() for n in (1,2,3)}
    for row in rows:
        # Each sentence is counted independently; never create EOS -> BOS edges.
        encoded=[vocabulary[t] for t in ['<s>',*row,'</s>']]
        for n in counts: counts[n].update(make_grams(encoded,n))
    probs={}
    for n in (2,3):
        totals=Counter()
        for gram,count in counts[n].items(): totals[gram[:-1]]+=count
        probs[n]={gram:count/totals[gram[:-1]] for gram,count in counts[n].items()}
    return vocabulary,counts,probs

def get_next_word(next_word_probs,n,start_text,*,vocabulary,rng=None,biprobs=None):
    if not isinstance(n,int) or isinstance(n,bool) or n<0: raise ValueError('n must be a nonnegative integer')
    rng=rng or random.Random(0)
    encoded=[vocabulary.get(t,vocabulary['<unk>']) for t in start_text]
    if not encoded: encoded=[vocabulary['<s>']]
    for _ in range(n):
        options={}
        # Longest context first, then bigram backoff; compare encoded IDs consistently.
        for table in (next_word_probs,biprobs or {}):
            for gram,p in table.items():
                if tuple(encoded[-(len(gram)-1):])==gram[:-1] and p>0:
                    options[gram[-1]]=p
            if options: break
        if not options: break  # unseen context terminates, no fabricated 'bad' token
        maximum=max(options.values())
        next_token=rng.choice(sorted(t for t,p in options.items() if p==maximum))
        if next_token==vocabulary['</s>']: break
        if next_token==vocabulary['<s>']: break
        encoded.append(next_token)
    return encoded

def generate(start_text='',n=10,*,vocabulary,probs,order=2,seed=0):
    if order not in (2,3): raise ValueError('order must be 2 or 3')
    ids=get_next_word(probs[order],n,['<s>',*tokenize(start_text)],vocabulary=vocabulary,
                      rng=random.Random(seed),biprobs=probs[2] if order==3 else None)
    inverse={i:t for t,i in vocabulary.items()}
    return ' '.join(inverse[i] for i in ids if i not in (vocabulary['<s>'],vocabulary['</s>']))
