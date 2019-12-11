
import sys

# changing words to lower case
def lowercase(infile, outfile):

    f = open(outfile, 'a')
    with open(infile) as fp:
        line = fp.readline()
        while line:
            row = line.split(" ")
            if row != []:
                row[0] = row[0].lower()
                output = " ".join(row)
                f.write(output)
            else:
                f.write("\n")
            line = fp.readline()
    f.close()


if __name__ == "__main__":
    print('changing document to lowercase...')
    lowercase(sys.argv[1], sys.argv[2])
    print('lowercasing done. new output file created.')