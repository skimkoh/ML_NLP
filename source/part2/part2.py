import sys

yx_lst = {}
y_lst = {}
x_lst = []
x_lst_distinct = []
x_cleaned = []

def emissionEstimate(filename):

    for line in open(filename, 'r'):
        line = line.rstrip()
        if line:
            line = line.rsplit(' ')
            x = line[0]
            y = line[1]
            x_lst.append(x)
            if y not in y_lst:
                y_lst[y] = 1
            else:
                y_lst[y] += 1
            if yx_lst.get((y, x)):
                yx_lst[(y, x)] += 1
            else:
                yx_lst[(y, x)] = 1
    e = {}
    for y, count_y in y_lst.items():
        for x in x_lst:
            if yx_lst.get((y, x)):
                e[(x, y)] = yx_lst[(y, x)] / count_y
            else:
                continue
    return e

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


def sentimentAnalysis(inputfile, e, outputfile):
    answers= []
    for line in open(inputfile, 'r'):
        line = line.rstrip()
        if line:
            line = line.rsplit(' ')
            x = line[0]
            x_copy = x
            if x not in x_cleaned:
                x_copy = "##UNK##"
            y_arg_max = 0
            tag_arg_max = ""
            for y in y_lst:
                if e.get((x_copy, y)):
                    y_arg = e[(x_copy, y)]
                else:
                    y_arg = 0
                if y_arg > y_arg_max:
                    y_arg_max = y_arg
                    tag_arg_max = y
            answers.append((x, tag_arg_max))
        else:
            answers.append(())
            
    with open(outputfile, 'w') as f:
        for i in answers:
            if i:
                word = i[0]
                label = i[1]
                f.write(f"{word} {label}\n")
            else:
                f.write(f"\n")

if __name__ == "__main__":
    print('training...')
    e = emissionEstimateSmoothing(sys.argv[1])
    print('training done. starting test...')
    sentimentAnalysis(sys.argv[2], e, sys.argv[3])
    print('test done.')


