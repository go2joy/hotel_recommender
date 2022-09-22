
import os
from utils.df_utils import *
from db.database import MyDatabase

from os.path import join
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils.df_utils import *
from db.sql_query import get_app_user, get_hotel, get_user_booking
pd.options.display.max_columns = None
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from recommender import HotelRecommender
# Importing dask dataframe
# import dask
# import dask.dataframe as dd
from sklearn.metrics.pairwise import cosine_similarity

# import ray
# ray.init(num_cpus=6, num_gpus=1, ignore_reinit_error=True)
# import modin.pandas as pd  #pip install modin[ray]

# db = MyDatabase()
# save_path = '/home/anhlbt/workspace/hotel_recommender/notebooks/data_recommender/'
save_path = '/home/ubuntu/workspace/hotel_recommender/notebooks/data_recommender/'
import time
from functools import reduce

from asyncio.log import logger
import sys
import time
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
from threading import Thread
import time
import re
from joblib import Parallel, delayed
import functools
import operator
import collections


class Apriori:
    minSupport = 0
    minConfidence = 0
    def __init__(self, Support, Confidence):
        self.minSupport = Support
        self.minConfidence = Confidence
        self.freqSet = defaultdict(int)
        self.localSet = defaultdict(int)
        
    def subsets(self,arr):
        """ Returns non empty subsets of arr"""
        return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])
    
    def get_transaction(self,item, transaction):
        # freqSet = defaultdict(int)
        localSet = defaultdict(int)
        if item.issubset(transaction):
            self.freqSet[item] += 1
            self.localSet[item] += 1
        # return localSet  
        # print(len(freqSet))    
    
    def returnItemsWithMinSupport(self,itemSet, transactionList, minSupport):
            """calculates the support for items in the itemSet and returns a subset
           of the itemSet each of whose elements satisfies the minimum support"""
            _itemSet = set()
            self.localSet = defaultdict(int)
            
            Parallel(n_jobs=6)(delayed(self.get_transaction)(item, transaction)\
                                                   for item in itemSet for transaction in transactionList)
            # df_res = pd.DataFrame(res)
            # df_res.columns = ['freqSet_dict','localSet_dict']
            # freqSet_dict,localSet_dict = list(df_res['freqSet_dict']), df_res['localSet_dict']
            # localSet_dict = df_res['localSet_dict']

            # freqSet = dict(functools.reduce(operator.add, map(collections.Counter, freqSet_dict)))
            # localSet = dict(functools.reduce(operator.add, map(collections.Counter, localSet_dict)))

#             for item in itemSet:
#                 for transaction in transactionList:
#                     if item.issubset(transaction):
#                         freqSet[item] += 1
#                         localSet[item] += 1
            # print("global ", len(freqSet))
            for item, count in self.localSet.items():
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
        # freqSet = defaultdict(int)
        largeSet = dict()
        # Global dictionary which stores (key=n-itemSets,value=support)
        # which satisfy minSupport

        assocRules = dict()
        # Dictionary which stores Association Rules

        oneCSet = self.returnItemsWithMinSupport(itemSet,
                                            transactionList,
                                            minSupport)

        currentLSet = oneCSet
        k = 2
        while(currentLSet != set([])):
            largeSet[k-1] = currentLSet
            currentLSet = self.joinSet(currentLSet, k)
            currentCSet = self.returnItemsWithMinSupport(currentLSet,
                                                    transactionList,
                                                    minSupport)
            currentLSet = currentCSet
            k = k + 1

        def getSupport(item):
                """local function which Returns the support of an item"""
                return float(self.freqSet[item])/len(transactionList)

        toRetItems = []
        for key, value in largeSet.items():
            toRetItems.extend([(tuple(item), getSupport(item))
                               for item in value])

#         Parallel(n_jobs=2)(delayed(sqrt_func)(i, j) for i in range(5) for j in range(2))
        

        toRetRules = []
#         for key, value in largeSet.items():
#             for item in value:
#                 _subsets = map(frozenset, [x for x in self.subsets(item)])
#                 for element in _subsets:
#                     remain = item.difference(element)
#                     if len(remain) > 0:
#                         confidence = getSupport(item)/getSupport(element)
#                         if confidence >= minConfidence:
#                             toRetRules.append(((tuple(element), tuple(remain)), confidence))
#         return toRetItems, toRetRules
    
        toRetRules = Parallel(n_jobs=6)(delayed(self._func)(item) for key, value in largeSet.items() for item in value)
        toRetRules = [item[0] for item in toRetRules]
        print(toRetItems)
        print(toRetRules)
        return  toRetItems, toRetRules
    
    def _func(self, item):
        global toRetRules
        _subsets = map(frozenset, [x for x in self.subsets(item)])
        for element in _subsets:
            remain = item.difference(element)
            if len(remain) > 0:
                confidence = getSupport(item)/getSupport(element)
                if confidence >= minConfidence:
                    toRetRules.append(((tuple(element), tuple(remain)), confidence))
        return toRetRules
    

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
                    
        
def run_apriori():
    recommender = HotelRecommender(n_samples=-1)

    test = Apriori(0.0005, 0.1)
#     inFile = test.dataFromFile('sparse_df.txt')
    inFile = test.dataFromDataFrame(recommender.user_lst_hotel)
    items, rules = test.runApriori(inFile,test.minSupport,test.minConfidence)
    test.saveResults(items, rules)
    
    tmp = pd.DataFrame(rules)
    tmp1 = pd.DataFrame(list((tmp[0].values)))
    tmp1.columns = ['A', 'B']
    tmp2 = pd.pivot_table(tmp1, index='A', aggfunc=lambda x: ' '.join(str(v) for v in x))
    tmp2 = tmp2.reset_index()
    tmp2.B = tmp2['B'].apply(lambda row: [int(float(x)) for x in re.findall(r"\d+\.?\d*", str(row))])
    #     add
    tmp2.B = tmp2['B'].apply(lambda row: list(set(row)))
    tmp2.A = tmp2['A'].apply(lambda row: sorted(list(int(float(x)) for x in row)))
    return tmp2
    
if __name__=="__main__":
    recommender = HotelRecommender(n_samples=-1)

    test = Apriori(0.0005, 0.01)
    inFile = test.dataFromDataFrame(recommender.user_lst_hotel)
    items, rules = test.runApriori(inFile,test.minSupport,test.minConfidence)

    test.saveResults(items, rules)

    tmp = pd.DataFrame(rules)
    tmp1 = pd.DataFrame(list((tmp[0].values)))
    tmp1['confidence'] = tmp[1].values
    tmp1.columns = ['A', 'B', 'confidence']
    tmp1 = tmp1.query('confidence > 0.2')
    tmp2 = pd.pivot_table(tmp1, index='A', aggfunc=lambda x: ' '.join(str(v) for v in x))
    tmp2 = tmp2.reset_index()
    tmp2.B = tmp2['B'].apply(lambda row: [int(float(x)) for x in re.findall(r"\d+\.?\d*", str(row))])
    #     add
    tmp2.B = tmp2['B'].apply(lambda row: list(set(row)))
    tmp2.A = tmp2['A'].apply(lambda row: sorted(list(int(float(x)) for x in row)))