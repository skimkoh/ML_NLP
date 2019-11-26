import sys
import os
import os.path


yx_lst = {}
y_lst = {}
x_lst = []
x_lst_distinct = []
x_cleaned = []

# taken from part 2

def emissionEstimateSmoothing(filename, k=3):
    
    for line in open(filename, 'r'):
        line = line.rstrip()
        if line:
            line = line.rsplit(' ')
            x = line[0]
            y = line[1]
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
                
    for y in y_lst:
        yx_lst[(y, '##UNK##')] = 0
        
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

def sentimentAnalysis(inputfile, e, q, outputfile, x_cleaned):
    if os.path.exists(outputfile):
        os.remove(outputfile)

    sentence = []
    for line in open(inputfile, 'r'):
        line = line.rstrip()
        if line:
            sentence.append(line)
        else:
            alpha_lst = alpha(sentence, e, q, x_cleaned, y_lst)
            beta_lst = beta(sentence, e, q, x_cleaned, y_lst)
            pred = []
            for i in range(len(sentence)):
                result = [0.0, '']
                for y in y_lst:
                    score = alpha_lst[i][y] * beta_lst[i][y]
                    if score > result[0]:
                        result = [score, y]
                pred.append(result[1])
    
            with open(outputfile, 'a') as f:
                for i in range(len(sentence)):
                        word = sentence[i]
                        label = pred[i]
                        if label != '':
                            f.write(f"{word} {label}\n")
                        else:
                            f.write(f"{word} O\n")          
                f.write(f"\n")  
                    
            sentence = []


def alpha(sentence, e, q, x_cleaned, y_lst):
    values = []

    # initialize each word in the sentence and assign the labels to be all 0 first
    for word in sentence:   
        labels = {}
        for y in y_lst:
            labels[y] = 0.0
        values.append(labels)


    for y in y_lst:
        if q.get((y, 'START')):
            values[0][y] = q[(y, 'START')]
    
    # recursion
    for i in range(1, len(sentence)):
        for current in y_lst:
            for previous in y_lst:
                if q.get((current, previous)):
                    if sentence[i-1] in x_cleaned:
                       if e.get((sentence[i-1], previous)):
                           values[i][current] += values[i-1][previous] * q[(current, previous)] * e[(sentence[i-1], previous)]
                    else:
                        if e.get(('##UNK##', previous)):
                            values[i][current] += values[i-1][previous] * q[(current, previous)] * e[('##UNK##', previous)]
    return values

    
def beta(sentence, e, q, x_cleaned, y_lst):
    values = []

    for word in sentence:   
        labels = {}
        for y in y_lst:
            labels[y] = 0.0
        values.append(labels)

    for y in y_lst:
        if q.get(('STOP', y)):
            if sentence[-1] in x_cleaned:
                if e.get((sentence[-1], y)):
                    values[-1][y] = q[('STOP', y)] * e[(sentence[-1], y)]
            else:
                if e.get(('##UNK##', y)):
                    values[-1][y] = q[('STOP', y)] * e[('##UNK##', y)]

    # recursion 
    for i in reversed(range(len(sentence)-1)):
        for previous in y_lst:
            for current in y_lst:
                if q.get((current, previous)):
                    if sentence[i] in x_cleaned:
                        if e.get((sentence[i], previous)):
                            values[i][previous] += values[i+1][current] * q[(current, previous)] * e[(sentence[i], previous)]
                    else:
                        if e.get(('##UNK##', previous)):
                            values[i][previous] += values[i+1][current] * q[(current, previous)] * e[('##UNK##', previous)]

    return values

if __name__ == "__main__":
    print('training....')
    e = emissionEstimateSmoothing(sys.argv[1])
    t = transitionEstimate(sys.argv[1])
    print('training done. starting test...')
    sentimentAnalysis(sys.argv[2], e, t, sys.argv[3], x_cleaned)
    print('test done.')

