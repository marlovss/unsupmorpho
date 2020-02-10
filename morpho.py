#!/usr/bin/python
import pickle
import re,nltk

import math
import numpy as np
import unicodedata
import codecs
import argparse
from nltk.tokenize.regexp import RegexpTokenizer



class PtgTokenizer(RegexpTokenizer):
    """
    Tokenize the given sentence in Portuguese.
    :param text: text to be tokenized, as a string
    """
    def __init__(self):
       tokenizer_regexp = r'''(?ux)
              # the order of the patterns is important!!
              # more structured patterns come first
              [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+|    # emails
              (?:https?://)?\w{2,}(?:\.\w{2,})+(?:/\w+)*|                  # URLs
              (?:[\#@]\w+)|                     # Hashtags and twitter user names
              (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
              (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
              (?:\B-)?\d+(?:[:.,]\d+)*(?:-?\w)*|
              # numbers in format 999.999.999,999, possibly followed by hyphen and alphanumerics
              # \B- avoids picks as F-14 as a negative number
              \.{3,}|                           # ellipsis or sequences of dots
              \w+|                              # alphanumerics
              -+|                               # any sequence of dashes
              \S                                # any non-space character
       '''
       RegexpTokenizer.__init__(self,tokenizer_regexp)

def normalize(list_of_words):
   return [unicodedata.normalize("NFKC", word.lower()) for word in list_of_words]
   

def load_words_corpus(list_of_corpora):
   wordHash = dict()
   invWordHash = dict() 
   for corpus in list_of_corpora:
      words = normalize(list(corpus.words())[:300])
      inverse = [word[::-1] for word in words] 
      for word in words:
         if word[0] not in wordHash:
            wordHash[word[0]]=[word]
         else:
            wordHash[word[0]].append(word)
      for word in inverse:
         if word[0] not in invWordHash:
            invWordHash[word[0]]=[word]
         else:
            invWordHash[word[0]].append(word)
   return words,wordHash,invWordHash

def load_words_files(file_list, encoding="utf-8"):
   wordHash = dict()
   invWordHash = dict() 
   tokenizer= PtgTokenizer()
   for filename in file_list:
      text = codecs.open(filename,encoding=encoding).read().lower()
      words = normalize(tokenizer.tokenize(text))
      inverse = [word[::-1] for word in words] 
      for word in words:
         if word[0] not in wordHash:
            wordHash[word[0]]=[word]
         else:
            wordHash[word[0]].append(word)
      for word in inverse:
         if word[0] not in invWordHash:
            invWordHash[word[0]]=[word]
         else:
            invWordHash[word[0]].append(word)
   return words,wordHash,invWordHash

def load_words_bin(words_name, wordHash_name, invWordHash_name):
   words = pickle.load(file(words_name_name,"b"))
   wordHash = pickle.load(file(wordHash_name,"b"))
   invWordHash = pickle.load(file(invWordHash_name,"b"))
   return words,wordHash,invWordHash



def dump_data(words,wordHash,invWordHash,suffixes,radicals, dirname):
  pickle.dump(words, open(os.join(dirname,"words.bin"),"wb"))
  pickle.dump(invWordHash, open(os.join(dirname,"invWordHashDic.bin"),"wb"))
  pickle.dump(wordHash, open(os.join(dirname,"wordHashDic.bin"),"wb"))
  pickle.dump(suffixes, open(os.join(dirname,"suffixes.bin"),"wb"))
  pickle.dump(radicals, open(os.join(dirname,"radicals.bin"),"wb"))



def compute_suf_rad(words, wordHash, invWordHash):
   lengths = [len(word) for word in words]
   maxLen = max(lengths)
   medLen = np.median(lengths)
   suffixCand= dict()
   for i in range(1,int(math.floor(medLen))):
      for term in invWordHash:
         for word in set(invWordHash[term]):
            suf = word[:i]
            if suf in suffixCand:
                continue
            else:
                similarSuffix= [w[::-1] for w in set(invWordHash[term]) if w[:i]==suf and w[i:]!=""]
                if len(set(similarSuffix))>1:
                    suffixCand[suf[::-1]]=similarSuffix
   suffixes = dict()
   radicals = dict()
   for key in suffixCand:
      sameSuffix = suffixCand[key]
      for word in sameSuffix:
         radical = word[:-len(key)]
         flections = [w for w in wordHash[word[0]] if w[:len(radical)] == radical]
         if len(set(flections)) >1:
             if key in suffixes:
                suffixes[key][radical]=flections
             else:
                suffixes[key]={radical:flections}
             if radical in radicals:
                if key in radicals[radical]:
                   radicals[radical][key].append(word)
                else:
                   radicals[radical][key]=[word]
             else:
               radicals[radical]={suf:[word]}
   return suffixes,radicals


def listToTxt(elem_list):
   txt=""
   size = len(elem_list)
   for i in range(size):
      txt+=elem_list[i]
      if i<size-1:
        txt+=", "
   return txt

def main():

   parser = argparse.ArgumentParser()
   parser.add_argument('--demo', action="store_true",
     help="Demo with Machado Corpus.")
   parser.add_argument('--dir', default=False, action="store_true",
     help="If the input is a corpus split into several files.")
   parser.add_argument('--enc', type=str, default="utf-8",
     help="If the input is a corpus split into several files.")
   parser.add_argument('--dump', action="store_true",
     help="Create bynary dumps of the word hashes and suffix lists for future use.")
   parser.add_argument('--outdir', type=str, default="./",
     help="Path to the directory where the dumps will be stored.")
   parser.add_argument("-q", "--quiet", default=False, action="store_true",
     help="Quiet mode -  does not print results in the standard output.")
   parser.add_argument('--path', type=str, default="./corpus.txt",
     help="Path to the file/directory for the input corpus.")

   FLAGS = parser.parse_args()

   words,wordHash,invWordHash=None,None,None
   if FLAGS.demo:
      from nltk.corpus import machado
      words,wordHash,invWordHash = load_words_corpus([machado])
   elif FLAGS.dir:
      words,wordHash,invWordHash = load_words_files(os.listdir(FLAGS.path),FLAGS.enc)
   else:
      words,wordHash,invWordHash = load_words_files([FLAGS.path],FLAGS.enc)

   suffixes,radicals = compute_suf_rad(words, wordHash, invWordHash)

   if FLAGS.dump:
     dump_data(words,wordHash,invWordHash,suffixes,radicals,FLAGS.outdir)
   print("teste")
   if not FLAGS.quiet:
     print("Number of suffixes: %d"%len(suffixes))
     for key in suffixes:
         radsTxt = listToTxt(list(suffixes[key].keys()))
         print("Suffix: %s \t Radicals: %s"%(key,radsTxt))




if __name__ == "__main__":
    main()    



