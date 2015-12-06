from __future__ import print_function

import sys,time
from operator import add

from pyspark import SparkContext


if __name__ == "__main__":

    sc = SparkContext(appName="PythonWordCount")
    #lines = sc.textFile(sys.argv[1], 1)
    exlist = [('a',1),('b',2),('a',3),('b',1),('c',2),('d',4),('c',8)]
    counts = sc.parallelize(exlist)
    counts = counts.combineByKey((lambda x: (x,1)),
                             (lambda x, y: (x[0] + y, x[1] + 1)),
                             (lambda x, y: (x[0] + y[0], x[1] + y[1])))
    output = counts.collect()
    dic = {}
    for (word, (a,b)) in output:
        dic[word] = float(a)/float(b)
        print("%s: %f" % (word, dic[word]))
    sc.stop()