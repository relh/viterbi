#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np

# Richard Higgins
# relh

class Conditional(object):
  def __init__(self):
    self.counts = {}
    self.probs = {}

  def count_put(self, cls, item): # tag tran
    self.insert(cls, item)

  def insert(self, cls, item):
    if cls in self.counts:
      if item in self.counts[cls]:  
        self.counts[cls][item] += 1
      else:
        self.counts[cls][item] = 1
    else:
      self.counts[cls] = {item: 1}

  def add_one(self, cls):
      # Add-one smoothing
      N = {}
      total = 0
      for count in range(0, max(self.counts[cls].values())+1):
        N[count] = 1 
      for count in self.counts[cls].values():
        total += count 
        N[count] += 1
      return N, total

  def calc_probs(self): # calc_tag_probs
    for cls in self.counts:    
      self.probs[cls] = {}

      N, total = self.add_one(cls)

      r_star = {}
      if 1 in N:
        r_star[0] = N[1]
      for num, count in N.items():
        if num+1 in N and num <= 5: # set k > 5 -> basic approximation
          r_star[num] = (num+1) * (N[num+1] / N[num])
        else:
          r_star[num] = num 

      for idx in r_star:
        self.probs[cls][idx] = r_star[idx] / total # / sum(N.values())

def build(f_h):
  for line in f_h:
    line = line.lower()
    final_words = []
    final_pos = []
    units = line.split(' ')
    first = 1
    for chunk in units:
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
          final_words.append(word)
          final_pos.append(pos)
          tags.insert(pos, word)
          trans.insert(old_pos, pos)
          old_pos = pos
          word, pos = part2.split('/')
        # Check for lots of /'s
        elif len(chunk.split('/')) > 2:
          pass # broken dataset
        else:
          word, pos = chunk.split('/')
        # Collect words
        final_words.append(word)
        final_pos.append(pos)
        words[line] = final_words
        poses[line] = final_pos
        # Collect pos
        tags.insert(pos, word)
        if first == 1:
          # Start of sentence
          trans.insert('start', pos)     
          first = 0        
        else:
          trans.insert(old_pos, pos)
        old_pos = pos

  tags.calc_probs()
  trans.calc_probs()

tags = Conditional() 
trans = Conditional() 
words = {}
poses = {}

#Let T = # of part-of-speech tags
#Build words too
#with open('POS.test') as test:
#  build(test)

with open('POS.train') as train:
  build(train)

# Base prediction model
#print trans.counts.values()
#print tags.counts.values()

def assess(model, path='POS.train'):
  correct = 0
  total = 0
  with open(path) as fh:
    for line in fh:
      line = line.lower()
      parse_words = words[line]
      parse_poses = poses[line]

      seq = []
      for word in parse_words:
        seq.append(model.predict(word))

      for i, v in enumerate(seq):
        if v == parse_poses[i]:
          correct += 1
        total += 1

  return correct / total

class Model(object):
  def __init__(self):
    pass

  def predict(self, word):
    pass

class Baseline(Model):
  def __init__(self):
    pass

  def predict(self, word):
    max_tag = 0
    max_counts = 0
    total = 0
    for tag in tags.counts:
      if word in tags.counts[tag]:
        total += tags.counts[tag][word]
        if tags.counts[tag][word] > max_counts:
           max_counts = tags.counts[tag][word]
           max_tag = tag
    #print max_counts/total
    return max_tag

class Viterbi(Model):
  def __init__(self):
    pass

  def predict(self, word):
    max_tag = 0
    max_counts = 0
    total = 0
    for tag in tags.counts:
      if word in tags.counts[tag]:
        total += tags.counts[tag][word]
        if tags.counts[tag][word] > max_counts:
           max_counts = tags.counts[tag][word]
           max_tag = tag
    #print max_counts/total
    return max_tag

baseline_score = assess(Baseline())
print baseline_score
viterbi_score = assess(Viterbi())
print viterbi_score 

# Here we iterate on a sentence by sentence basis
with open('POS.train') as train:
  for line in train:
    line = line.lower()
    #W = # of words in the sentence
    W = len(words[line])

    #/* Initialization Step */
    sentence = words[line]
    score = {}
    backptr = {}

    #for t = 1 to T
    for t in tags:
      score[t] = []
      backptr[t] = []
      #Score(t, 1) = Pr(W1| Tt) * Pr(Tt| φ) # Pr(t | start)
      w_count = tags[t][sentence[0]] if sentence[0] in tags[t] else 0
      t_count = trans['start'][t] if t in trans['start'] else 0
      #print tags_probs[t]
      pr_w_t = tags_probs[t][w_count]
      pr_t_s = trans_probs['start'][t_count] 
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
        max_score = -1
        for i, ti in enumerate(tags):
          t_count = trans[ti][t] if t in trans[ti] else 0
          next_score = score[ti][-1] * trans_probs[ti][t_count]

          if next_score > max_score:
            max_index = i
            max_key = ti
            max_score = next_score
            print max_score

        t_count = trans[max_key][t] if t in trans[max_key] else 0
        pr_t_ti = trans_probs[max_key][t_count]

        score[t].append(pr_w_t * pr_t_ti)

        #BackPtr(t, w) = index of j that gave the max above
        backptr[t].append(max_key) #max_index) 

    #/* Sequence Identification */
    #Seq(W ) = t that maximizes Score(t,W )
    seq = []
    max_index, max_key = max(enumerate(tags.keys()), key=lambda p: score[p[1]][-1]) # Need Prob transitions between tags
    seq.append(max_key) #score = np.argmax(score[i]) 
    #for w = W -1 to 1
    for j in range(1, W-1):
      #Seq(w) = BackPtr(Seq(w+1),w+1)
      print seq
      print backptr[seq[0]]
      seq.insert(0, backptr[seq[0]][W-j]  )

    print line
    print sentence
    print seq
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
