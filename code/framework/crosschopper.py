# This script takes in a file of data, organized by lines, and chops it up into training and test sets.

import sys

args = sys.argv
if len(args) < 4:
    print("This script chops a data file into pieces for cross validation.")
    print("Each data element should be on a single line. This script uses rotary segmentation.")
    print("Usage: python3",args[0],"<infile.dat> <outprefix> <#sets>")
    quit(1)

numchops = int(args[3])
outprefix = args[2]

training_files = list()
test_files = list()
# open all our output files
for i in range(numchops):
    training_files.append(open(outprefix + "."+str(i)+"-"+str(numchops)+".training.dat","w"))
    test_files.append(open(outprefix + "."+str(i)+"-"+str(numchops)+".test.dat","w"))

infile = open(args[1],"r")

for num,line in enumerate(infile):
    mod = num % numchops
    for i in range(numchops):
        if i == mod:
            test_files[i].write(line)
        else:
            training_files[i].write(line)

infile.close()

for i in range(numchops):
    training_files[i].close()
    test_files[i].close()
