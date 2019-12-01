import sys
import os
import os.path
yx_lst = {}   # counting the y -> x, where the observation seq is generated by the tag
y_lst = {}    # counting the occurrences of the tags in the document
x_lst = []    # showing all the words in the document
x_lst_distinct = []     # remove duplicates words. Is used to extract words that appear > 3
x_cleaned = []   # store all the words in the training set that is not 'UNK'. Is used to compare with testing set


# taken from part 2. used for viterbi algorithm
def emissionEstimateSmoothing(filename, k=3):
    
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
def sentimentAnalysis(inputfile, e, t, outputfile, x_cleaned):

    if os.path.exists(outputfile):
        os.remove(outputfile)

    sentence = []
    for line in open(inputfile, 'r'):
        line = line.rstrip()
        if line:
            sentence.append(line)
        else:
            output = viterbi(sentence, e, t, x_cleaned, y_lst)  
            with open(outputfile, 'a') as f:
                for i in range(len(sentence)):
                    word = sentence[i]
                    label = output[i]
                    if label != '':
                        f.write(f"{word} {label}\n")
                    else:
                        f.write(f"{word} 0\n")

                f.write(f"\n")
            sentence = []

def viterbi(sentence, e, t, x_cleaned, y_lst):
    
    pi = []
    for word in sentence:
        labels = {}
        for y in y_lst:
            labels[y] = [0.0, '']
        pi.append(labels)

    # first word in the sentence. initalize this to get the pi of the first word
    for y in y_lst: 
        if t.get((y, 'START')):
            if sentence[0] in x_cleaned:    # not unk
                if e.get((sentence[0], y)):  
                    em = e.get((sentence[0], y), 0)
                else:
                    em = 0.0
            else:
                if e.get(('##UNK##', y)):
                    em = e.get(('##UNK##', y), 0)
            
            pi[0][y] = [t[(y, 'START')] * em, 'START']

    # recursively forward
    for i in range(1, len(sentence)):
        for current in y_lst:
            if sentence[i] in x_cleaned:  # not unk
                if e.get((sentence[i], current)):
                    em = e.get((sentence[i], current),0)
                else:
                    em = 0.0

            else:    # unk 
                if e.get(('##UNK##', current)):
                    em = e.get(('##UNK##', current), 0)


            for previous in y_lst:
                if t.get((current, previous)):
                    c_score = pi[i-1][previous][0] * t.get((current, previous), 0) * em
                    if c_score > pi[i][current][0]:    # take the max for each pi
                        pi[i][current] = [c_score, previous]

    result = [0.0, '']
    for previous in y_lst:
        if t.get(('STOP', previous)):
            score = pi[-1][previous][0] * t.get(('STOP', previous), 0)
            if score > result[0]:
                result = [score, previous]
    
    if result[1]:
        pred = [result[1]]
        for j in reversed(range(len(sentence))):
            if j != 0:  
                pred.insert(0, pi[j][pred[0]][1]) 

    else:
        pred = ['']
        for j in reversed(range(len(sentence))):
            if j != 0:
                pred.insert(0, '')
    
    return pred

      
if __name__ == "__main__":
    print('training....')
    e = emissionEstimateSmoothing(sys.argv[1])
    t = transitionEstimate(sys.argv[1])
    print('training done. starting test...')
    sentimentAnalysis(sys.argv[2], e, t, sys.argv[3], x_cleaned)
    print('test done.')
