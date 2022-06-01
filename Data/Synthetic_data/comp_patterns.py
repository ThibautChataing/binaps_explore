import argparse
import os
import pandas as pd
import numpy as np
import math



def readBinapsPats(p):

    with open(p) as patF:

        ls = patF.readlines()
        # drop enclosing brackets, offset by 1, remove additional whitespaces
        pats = [ tuple([int(x)+1 for x in l[1:-2].split(" ") if x != '']) for l in ls]
        # make patterns unique
        pats = set(pats)
        return pats

def readAssoPats(p):

    with open(p) as patF:

        ls = patF.readlines()
        pats = [ tuple([int(x) for x in l[:-1].split(" ")]) for l in ls]
        return set(pats)

def readDescPats(p):

    with open(p) as patF:

        ls = patF.readlines()
        pats = [ tuple([int(x) for x in l[1:-2].replace(',', '').split(" ") if x != '']) for l in ls]
        return set(pats)

def readSlimPats(p, dbFilepath):

    dbFile = open(dbFilepath)
    l = dbFile.readline()
    l = dbFile.readline()
    l = dbFile.readline()
    l = dbFile.readline()
    l = dbFile.readline()
    l = dbFile.readline()[4:-1]
    dbmap = [int(x) for x in l.split(" ")]
    dbFile.close()
    with open(p) as patF:

        ls = patF.readlines()
        # drop header
        ls = ls[2:]
        # drop laste entry which gives usage
        pats = [ tuple(sorted([dbmap[int(x)] for x in l[:-1].split(" ")[:-3]])) for l in ls]
        return set(pats)

def refPatsParser(r):

    with open(r) as patF:

        ls = patF.readlines()
        pats = [ tuple(sorted([int(x) for x in l[:-1].split(" ")])) for l in ls]
        return set(pats)


# Compare two pattern sets p1 and p2 using Jaccard distance
def compJaccard(p1, p2):

    return len(p1.intersection(p2))/ len(p1.union(p2))

# Compare two pattern sets p1 and p2 using F1 score (harmonic mean of precision and recall)
# F1 = TP / (TP + 1/2(FP + FN))
def compF1(p1, p2):

    interCount = len(p1.intersection(p2))
    return interCount / (interCount + .5*len(p1.symmetric_difference(p2)))





def main():
    parser = argparse.ArgumentParser(description='.dat to .hdf5 parser')
    parser.add_argument('-p','--predicted', required=True,
                        help='Pattern set file yielded by algorithm')
    parser.add_argument('-t', '--type', required=True,
                        help='Algorithm type (Binaps/Asso/Desc/Slim)')
    parser.add_argument('-r', '--reference', required=True,
                        help='Reference ground truth pattern set')
    parser.add_argument('-m', '--metric', required=True,
                        help='Metric to compare with (Jaccard/F1)')
    args = parser.parse_args()

    #drop suffix and add db
    dbFilepath = args.reference[:-13] + '.db'

    algoTypeSwitch = {
        "Binaps" : readBinapsPats,
        "Asso" : readAssoPats,
        "Desc" : readDescPats,
        "Slim" : lambda x : readSlimPats(x, dbFilepath)
    }
    predPatsParser = algoTypeSwitch.get(args.type, lambda: "Unknown algorithm for given pattern file. Specify known algorithm type using the -t option.")


    p_pred = predPatsParser(args.predicted)

    p_ref = refPatsParser(args.reference)

    metricTypeSwitch = {
        "Jaccard" : compJaccard,
        "F1" : compF1
    }
    compPats = metricTypeSwitch.get(args.metric, lambda: "Unknown metric specified. Specify known metric using the -m option")

    print(compPats(p_pred, p_ref))



if __name__ == '__main__':
    main()