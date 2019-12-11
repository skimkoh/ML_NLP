import sys
import os
import os.path
from copy import deepcopy
import copy
import numpy as np
yx_lst = {}   # counting the y -> x, where the observation seq is generated by the tag
y_lst = {}    # counting the occurrences of the tags in the document
x_lst = []    # showing all the words in the document
x_lst_distinct = []     # remove duplicates words. Is used to extract words that appear > 3
x_cleaned = []   # store all the words in the training set that is not 'UNK'. Is used to compare with testing set

# taken from part 2. used for viterbi algorithm


# k is changed to 2
def emissionEstimateSmoothing(filename, k=2):
    
    for line in open(filename, 'r'):
        line = line.rstrip()
        if line:
            line = line.rsplit(' ')
            x = line[0]     # the word
            y = line[1]     # the tag
            x_lst.append(x)
            if x not in x_lst_distinct:
                x_lst_distinct.append(x)
            if y not in y_lst:
                y_lst[y] = 1
            else:
                y_lst[y] += 1
            if yx_lst.get((y, x)):
                yx_lst[(y, x)] += 1
            else:
                yx_lst[(y, x)] = 1
                
    # initialise tag -> 'UNK' for all tags     
    for y in y_lst:
        yx_lst[(y, '##UNK##')] = 0
        
    # create a list that replaces words that appear < k with 'UNK'. 
    # rest of the words remain the same and is added to the list. 

    replace = []
    for x in x_lst_distinct:
        summa = 0
        for y in y_lst:
            if yx_lst.get((y, x)):
                summa += yx_lst[(y, x)]
        if summa >= k:
            replace.append(x)
        else: 
            replace.append('##UNK##')
            for y in y_lst.keys():
                if yx_lst.get((y, x)):
                    yx_lst[(y, '##UNK##')] += yx_lst.pop((y, x), 0)
                    
    for i in replace: 
        if i == "##UNK##":
            continue
        else:
            x_cleaned.append(i)
        
    e = {}
    for y, count_y in y_lst.items():
        for x in replace:
            if yx_lst.get((y, x)):
                c = yx_lst[(y, x)]
                if c:
                    e[(x, y)] = c / count_y
    return e

# part 3 (i)
t_lst = {} # or u -> v
count_lst = {}  # occurrences we see the state 

def transitionEstimate(filename):
    
    count_lst['STOP'] = 0
    count_lst['START'] = 0
    u = 'START'

    for line in open(filename, 'r'):
        line = line.rstrip()
        if line:
            if u == 'START':
                count_lst['START'] += 1
            line = line.rsplit(' ')
            v = line[1]
            if v not in count_lst:
                count_lst[v] = 1
            else:
                count_lst[v] += 1
            if t_lst.get((u, v)):
                t_lst[(u, v)] += 1
            else:
                t_lst[(u, v)] =1
            u = v
        else: 
            count_lst['STOP'] += 1
            if t_lst.get((u, 'STOP')):
                t_lst[(u, 'STOP')] += 1
            else:
                t_lst[(u, 'STOP')] = 1
                
            u = 'START'
            
    e = {}
    for yi_1, count in count_lst.items():
        for yi in count_lst:
            if t_lst.get((yi_1, yi)):
                e[(yi, yi_1)] = t_lst[(yi_1, yi)] / count
            else:
                continue
    return e


# part 3 (ii)
big = -9999999
def sentimentAnalysis(inputfile, e, t, outputfile, x_cleaned_dis):
    
    if os.path.exists(outputfile):
        os.remove(outputfile)

    sentence = []
    for line in open(inputfile, 'r'):
        line = line.rstrip()
        if line:
            sentence.append(line)
        else:
            output = viterbi(sentence, e, t, x_cleaned_dis, y_lst)
            with open(outputfile, 'a') as f:
                for i in range(len(sentence)):
                    if output == None: 
                          continue
#                         word = sentence[i]
#                         f.write(f"{word} O\n")
                    else:
                        word = sentence[i]
                        label = output[i]
                        f.write(f"{word} {label}\n")

                f.write(f"\n")
            
            sentence = []


def viterbi(sentence, e, t, x_cleaned, y_lst):

    pi = []
    for word in sentence:
        labels = {}
        for y in y_lst:
            labels[y] = [big, '']

        pi.append(labels)

    sentence2 = deepcopy(sentence)
    
    if sentence2[0] not in x_cleaned:
        sentence2[0] = '##UNK##'
        
    for y in y_lst:
        ttt = 0
        eee = 0
        if t.get((y, 'START')):
            ttt = np.log(t.get((y, 'START'))) 
        else: 
            ttt = -99999
        if e.get((sentence2[0], y)):
            eee =  np.log(e.get((sentence2[0], y)))
        else:
            eee = -99999
        value = ttt + eee
        if value > pi[0][y][0]:
            pi[0][y] = [value, 'START']


    for k in range(1, len(sentence)):
        if sentence2[k] not in x_cleaned:
            sentence2[k] = '##UNK##'
        word = sentence2[k]
        
        for v in y_lst:
            temp = []
            for u in y_lst:
                ttt = 0
                eee = 0
                if t.get((v, u)):
                    ttt = np.log(t.get((v, u)))
                else:
                    ttt = -99999
                if e.get((word, v)):
                    eee = np.log(e.get((word, v)))
                else:
                    ttt = -99999
                value = pi[k-1][u][0] + ttt + eee
                if value > pi[k][v][0]: 
                    pi[k][v] = [value, u]
            

    result = [big, '']
    for previous in y_lst:
        ttt = 0
        if t.get(('STOP', previous)):
            ttt = np.log(t.get(('STOP', previous)))
        else:
            ttt = -9999999
        
        score = pi[-1][previous][0] + ttt

        if score > result[0]:
            result = [score, previous]
    
    pred = [result[1]]
    for j in reversed(range(len(sentence))):
        if j == 0: break
        pred.insert(0, pi[j][pred[0]][1]) 

    return pred

  
if __name__ == "__main__":
    print('training....')
    e = emissionEstimateSmoothing(sys.argv[1])
    t = transitionEstimate(sys.argv[1])
    print('training done. starting test...')
    sentimentAnalysis(sys.argv[2], e, t, sys.argv[3], x_cleaned)
    print('test done.')

