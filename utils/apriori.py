"""
Usage:
    $python apriori.py -f DATASET.csv -s minSupport  -c minConfidence
    $python apriori.py -f DATASET.csv -s 0.15 -c 0.6
"""
import os
os.environ["MODIN_ENGINE"] = "dask" 
from asyncio.log import logger
from calendar import c
import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from threading import Thread
import time
import pandas as pd
# import modin.pandas as pd 
# Importing dask dataframe
# import dask
# import dask.dataframe as pd
import re
from functools import reduce


class Apriori:
    minSupport = 0
    minConfidence = 0
    def __init__(self, Support, Confidence):
        self.minSupport = Support
        self.minConfidence = Confidence
        
        
    def subsets(self,arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])

    def returnItemsWithMinSupport(self,itemSet, transactionList, minSupport, freqSet):
            """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""
            _itemSet = set()
            localSet = defaultdict(int)

            for item in itemSet:
                    for transaction in transactionList:
                            if item.issubset(transaction):
                                    freqSet[item] += 1
                                    localSet[item] += 1

            for item, count in localSet.items():
                    support = float(count)/len(transactionList)

                    if support >= minSupport:#>=
                            _itemSet.add(item)

            return _itemSet


    def joinSet(self,itemSet, length):
            """Join a set with itself and returns the n-element itemsets"""
            return set([i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length])


    def getItemSetTransactionList(self,data_iterator):
        transactionList = list()
        itemSet = set()
        for record in data_iterator:
            transaction = frozenset(record)
            transactionList.append(transaction)
            for item in transaction:
                itemSet.add(frozenset([item]))              # Generate 1-itemSets
        return itemSet, transactionList


    def runApriori(self,data_iter, minSupport, minConfidence):
        """
        run the apriori algorithm. data_iter is a record iterator
        Return both:
         - items (tuple, support)
         - rules ((pretuple, posttuple), confidence)
        """
        itemSet, transactionList = self.getItemSetTransactionList(data_iter)

        freqSet = defaultdict(int)
        largeSet = dict()
        # Global dictionary which stores (key=n-itemSets,value=support)
        # which satisfy minSupport

        assocRules = dict()
        # Dictionary which stores Association Rules

        oneCSet = self.returnItemsWithMinSupport(itemSet,transactionList, minSupport,freqSet)

        currentLSet = oneCSet
        k = 2
        while(currentLSet != set([])):
            largeSet[k-1] = currentLSet
            currentLSet = self.joinSet(currentLSet, k)
            currentCSet = self.returnItemsWithMinSupport(currentLSet, transactionList, minSupport,freqSet)
            currentLSet = currentCSet
            k = k + 1

        def getSupport(item):
                """local function which Returns the support of an item"""
                return float(freqSet[item])/len(transactionList)

        toRetItems = []
        for key, value in largeSet.items():
            toRetItems.extend([(tuple(item), getSupport(item))
                               for item in value])

        toRetRules = []
        for key, value in largeSet.items():
            for item in value:
                _subsets = map(frozenset, [x for x in self.subsets(item)])
                for element in _subsets:
                    remain = item.difference(element)
                    if len(remain) > 0:
                        confidence = getSupport(item)/getSupport(element)
                        if confidence >= minConfidence:
                            toRetRules.append(((tuple(element), tuple(remain)),
                                               confidence))
        return toRetItems, toRetRules


    def printResults(self, items, rules):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        for item, support in sorted(items, key=lambda x: x[1]):
            print("item: %s , %.3f" % (str(item), support))
        print("\n------------------------ RULES:")
        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            print("Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence))


    def to_str_results(self, items, rules):
        """prints the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        i, r = [], []
        for item, support in sorted(items, key=lambda x: x[1]):
            x = "item: %s , %.3f" % (str(item), support)
            i.append(x)

        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            x = "Rule: %s ==> %s , %.3f" % (str(pre), str(post), confidence)
            r.append(x)

        return i, r
            
    def saveResults(self,items, rules):
        """save the generated itemsets sorted by support and the confidence rules sorted by confidence"""
        f = open('results.txt', 'w')
        for item, support in sorted(items, key=lambda x: x[1]):
            f.write("\nitem: %s , %.3f" % (str(item), support)) 
        f.write("\n------------------------ RULES:")
        i = 0
        for rule, confidence in sorted(rules, key=lambda x: x[1]):
            pre, post = rule
            i+=1
            f.write("\nRule %s : %s ==> %s , %.3f" % (str(i), str(pre), str(post), confidence)) 

    def dataFromFile(self,fname):
            """Function which reads from the file and yields a generator"""
            file_iter = open(fname, 'rU')
            for line in file_iter:
                    line = line.strip().rstrip(',')                         # Remove trailing comma
                    record = frozenset(line.split(','))
                    yield record

    def dataFromDataFrame(self,df):
        """Function which reads from the file and yields a generator"""
        for index, line in df.iterrows():                    # Remove trailing comma
                record = frozenset(line['booked_hotels'])
                yield record                    






