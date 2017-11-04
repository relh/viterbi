#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import argparse
import os

# Richard Higgins
# relh
lines = 0

def parse_arg(): # parses all the command line arguments
    parser = argparse.ArgumentParser('Viterbi')
    parser.add_argument('train_file', type=str, default='POS.train', help='training file')
    parser.add_argument('test_file', type=str, default='POS.test', help='test file')

    args = parser.parse_args()
    if not os.path.exists(args.train_file) or not os.path.exists(args.test_file):
        parser.error('Either your training or test file does not exist'.format())
    return args

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

  def build_N(self, cls, add_one=False):
      # Add-one smoothing
      N = {}
      total = 0
      if add_one:
        for count in range(0, max(self.counts[cls].values())+1):
          N[count] = 1 
      for count in self.counts[cls].values():
        total += count 
        if count in N:
          N[count] += 1
        else:
          N[count] = 1
      return N, total

  def old_calc_probs(self): # calc_tag_probs
    for cls in self.counts:    
      self.probs[cls] = {}

      N, total = self.build_N(cls)

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

      if 0 not in self.probs[cls]:
        self.probs[cls][0] = 0.0000000001

  def calc_probs(self): # calc_tag_probs
    for cls in self.counts:    
      self.probs[cls] = {}

      N, total = self.build_N(cls)

      r_star = {}
      if 1 in N:
        r_star[0] = N[1] / 171476 # Num of words in language (an approximation for unseen)
      else:
        r_star[0] = 1 / 171476 # Add one smoothing example word seen once
      for num, count in N.items():
        if num+1 in N and num <= 5: # set k > 5 -> basic approximation
          r_star[num] = (num+1) * (N[num+1] / N[num])
        else:
          r_star[num] = num 

      for idx in r_star:
        self.probs[cls][idx] = r_star[idx] / total # / sum(N.values())

def parse_chunk(chunk):
  # Skip end of sentences
  word = None
  pos = None
  if not '\n' in chunk:
    # Check for real \ case 
    if len(chunk.split('\/')) > 1:
      pos = chunk.split('/')[-1]
      word = '/'.join(chunk.split('/')[:-1])
    # Check for - and & words with a */. This is strange
    elif len(chunk.split('/')) > 2 and len(chunk.split('*/')) > 1:
      pass
      """
      if '&' in chunk:
        part1, part2 = chunk.split('&')
      elif '-' in chunk:
        part1, part2 = chunk.split('-')
      old_pos = parse_chunk(part1, old_pos)
      old_pos = parse_chunk(part2, old_pos)
      """
    # Check for lots of /'s
    elif len(chunk.split('/')) > 2:
      pass # broken dataset
    else:
      word, pos = chunk.split('/')
  return word, pos

def parse_line(line, train=True):
  line = line.lower()
  units = line.split(' ')

  final_words = []
  final_pos = []
  first = 1
  for chunk in units:
    word, pos = parse_chunk(chunk)
    if word is None:
      continue

    # Collect words
    final_words.append(word)
    final_pos.append(pos)
    if train:
      tags.insert(pos, word)
      if first == 1: # Start of sentence
        trans.insert('start', pos)     
        first = 0        
      else:
        trans.insert(old_pos, pos)
      old_pos = pos

  words[line] = final_words
  poses[line] = final_pos
     

def build(f_h):
  global lines
  for line in f_h:
    lines += 1
    parse_line(line)

  tags.calc_probs()
  trans.calc_probs()

class Model(object):
  def __init__(self):
    pass

  def predict(self, word):
    pass

class Baseline(Model):
  def __init__(self):
    pass

  def predict(self, line):
    seq = []
    for word in line:
      seq.append(self.predict_word(word))
    return seq

  def predict_word(self, word, pos=None):
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

    # Here we iterate on a sentence by sentence basis
    line = line.lower()
    #W = # of words in the sentence
    W = len(words[line])

    #/* Initialization Step */
    sentence = words[line]
    self.score = {}
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
  def predict(self, line):
      seq = []
      for i, word in enumerate(line):
        pos = seq[-1] if len(seq) > 0 else 'start'
        seq.append(self.predict_word(word, pos))
      return seq
      #print "TESTING " + parse_poses[i]
      #while True:
      #  pass

  def predict_word(self, word, pos):
    #for t = 1 to T
    max_score = 0
    max_tag = None
    for t in tags.counts.keys():
      #Score(t, 1) = Pr(W1| Tt) * Pr(Tt| φ) # Pr(t | start)
      w_count = tags.counts[t][word] if word in tags.counts[t] else 0
      t_count = trans.counts[pos][t] if t in trans.counts[pos] else 0

      pr_w_t = tags.probs[t][w_count]
      pr_t_s = trans.probs[pos][t_count] 
      #print pr_w_t
      #print pr_t_s
      if (pr_w_t * pr_t_s) > max_score:
        max_score = pr_w_t * pr_t_s
        max_pos = t
    return max_pos

def assess(model, path='POS.train'):
  correct = 0
  total = 0
  lines_done = 0
  with open(path) as fh:
    for line in fh:
      line = line.lower()

      if line not in words:
        parse_line(line, False)
      parse_words = words[line]
      parse_poses = poses[line]

      seq = model.predict(parse_words)

      for i, v in enumerate(seq):
        if v == parse_poses[i]:
          correct += 1
        total += 1

      lines_done += 1
      if lines_done % 1000 == 0:
        print "{} / {}".format(lines_done, lines)

  return correct / total

if __name__ == "__main__":
  args = parse_arg() # Parse paths to data files
  #Let T = # of part-of-speech tags
  #Build words too

  tags = Conditional() 
  trans = Conditional() 
  words = {}
  poses = {}

  with open(args.train_file) as train:
    build(train)

  print "Viterbi"
  viterbi_score = assess(Viterbi(), args.test_file)
  print viterbi_score 
  print "Baseline"
  baseline_score = assess(Baseline(), args.test_file)
  print baseline_score
