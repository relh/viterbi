#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# Richard Higgins
# relh

tags = {} 
trans = {}
words = {}


def tag(pos, word):
  if pos in tags:
    if word in tags[pos]:
      tags[pos][word] += 1
    else:
      tags[pos][word] = 1
  else:
    tags[pos] = {word: 1}


def tran(last, cur):
  if last+cur in trans:
    trans[last+cur] += 1
  else:
    trans[last+cur] = 1


def build(f_h):
  units = []
  for line in f_h:
    final_units = []
    units.append(line.split(' '))
    first = 1
    for chunk in units[-1]:
      # Skip end of sentences
      if not '\n' in chunk:
        # Check for real \ case 
        if len(chunk.split('\/')) > 1:
          pos = chunk.split('/')[-1]
          word = '/'.join(chunk.split('/')[:-1])
        # Check for - and & words with a */. This is strange
        elif len(chunk.split('/')) > 2 and len(chunk.split('*/')) > 1:
          if '&' in chunk:
            part1, part2 = chunk.split('&')
          elif '-' in chunk:
            part1, part2 = chunk.split('-')
          word, pos = part1.split('/')
          final_units.append(word)
          tag(pos, word)
          tran(old_pos, pos)
          old_pos = pos
          word, pos = part2.split('/')
        # Check for lots of /'s
        elif len(chunk.split('/')) > 2:
          pass # broken dataset
        else:
          word, pos = chunk.split('/')
        # Collect words
        final_units.append(word)
        words[line] = final_units
        # Collect pos
        tag(pos, word)
        if first == 1:
          # Start of sentence
          tran('start', pos)     
          first = 0        
        else:
          tran(old_pos, pos)
        old_pos = pos
  return tags, words

def get_tag_sum():
  pass

#Let T = # of part-of-speech tags
#Build words too
with open('POS.test') as test:
  build(test)

with open('POS.train') as train:
  build(train)

#print words
#print tags

def prob(d, one, two):
  prb = d[one+two]/sum([d[one+b] for b in tags if one+b in d]) if one+two in d else 0.0
  #print prb
  return prb

# Here we iterate on a sentence by sentence basis
with open('POS.train') as train:
  for line in train:
    #W = # of words in the sentence
    W = len(words[line])

    #/* Initialization Step */
    sentence = words[line]
    score = {}
    backptr = {}
    #print sentence

    #for t = 1 to T
    for t in tags:
      print t
      score[t] = []
      backptr[t] = []
      #Score(t, 1) = Pr(W1| Tt) * Pr(Tt| φ) # Pr(t | start)
      pr_w_t = tags[t][sentence[0]]/sum(tags[t].values()) if sentence[0] in tags[t] else 0.0
      pr_t_s = prob(trans, 'start', t) 
      score[t].append(pr_w_t * pr_t_s)

      #BackPtr(t, 1) = 0;
      backptr[t].append(0)

    #/* Iteration Step */
    #for w = 2 to W
    for w in sentence[1:]:
      #for t = 1 to T
      for t in tags:
        #Score(t, w) = Pr(Ww| Tt) *MAXj=1,T(Score(j, w-1) * Pr(Tt| Tj)) # Pr(t | ti)
        pr_w_t = tags[t][w]/sum(tags[t].values()) if w in tags[t] else 0.0
        max_index = 0
        max_key = ''
        max_score = 0
        for i, ti in enumerate(tags):
          next_score = score[ti][-1] * prob(trans, ti, t)
          print "GREAT SCOT"
          print (trans[ti+t]/sum([trans[tj+t] for tj in tags if tj+t in trans]) if ti+t in trans else 0.0)
          print prob(trans, ti, t)

          if next_score > max_score:
            print 'holla'
            max_index = i
            max_key = ti
            max_score = next_score
      
        print max_key
        pr_t_ti = trans[max_key+t]/sum([trans[max_key+tj] for tj in tags if max_key+tj in trans]) if max_key+t in trans else 0.0

        score[t].append(pr_w_t * pr_t_ti)
        #BackPtr(t, w) = index of j that gave the max above
        backptr[t].append(max_index) 

    while True:
      pass

    #/* Sequence Identification */
    #Seq(W ) = t that maximizes Score(t,W )
    seq = []
    max_index, max_key = max(enumerate(tags.keys()), key=lambda p: score[p[1]][-1]) # Need Prob transitions between tags
    seq.append(max_key) #score = np.argmax(score[i]) 
    #for w = W -1 to 1
    for j in range(1, W-1):
      #Seq(w) = BackPtr(Seq(w+1),w+1)
      print backptr#[seq[0]]
      seq.insert(0, backptr[seq[0]][W-j]  )

    break

'''

/* Initialization Step */
for t = 1 to T
Score(t, 1) = Pr(W1| Tt) * Pr(Tt| φ)
BackPtr(t, 1) = 0;

/* Iteration Step */
for w = 2 to W
for t = 1 to T
Score(t, w) = Pr(Ww| Tt) *MAXj=1,T(Score(j, w-1) * Pr(Tt| Tj))
BackPtr(t, w) = index of j that gave the max above

/* Sequence Identification */
Seq(W ) = t that maximizes Score(t,W )
for w = W -1 to 1
Seq(w) = BackPtr(Seq(w+1),w+1)

'''
