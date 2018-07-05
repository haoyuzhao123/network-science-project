import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

#util functions
def floatEqual(a,b):
    return np.abs(a-b) < np.exp(-20)

def findMax(vec):
    maxidx = 0
    maxnum = vec[0]
    for i in range(1,len(vec)):
        if vec[i] > maxnum:
            maxnum = vec[i]
            maxidx = i
    return [maxnum,maxidx]

def findMin(vec):
    minidx = 0
    minnum = vec[0]
    for i in range(1,len(vec)):
        if vec[i] < minnum:
            minnum = vec[i]
            minidx = i
    return [minnum,minidx]

class Edge:
    def __init__(self):
        self.INF = -10000
        self.weight = 0
        self.nodein = -1
        self.nodeout = -1
        self.visit = 0
        self.empmean = -self.INF
        self.lastvalue = 0
        self.lowerconfidence = 0

    def setWeight(self, w):
        self.weight = w
    
    def setNodes(self, v, u):
        self.nodein = u
        self.nodeout = v

    def setVisit(self, n):
        self.visit = n

    def oneVisit(self):
        self.visit += 1

class Graph:
    def __init__(self):
        self.adjlist = []
        self.numnodes = 0

    def setNumnodes(self, n):
        self.numnodes = n
        self.adjlist = [[]] * n
    
    def addEdge(self, e):
        nout = e.nodeout
        self.adjlist[nout].append(e)
