class Node:
    def __init__(self):
        self.name = -1

    def setName(self, name):
        self.name = name

class Edge:
    def __init__(self):
        self.weight = 0
        self.nodein = -1
        self.nodeout = -1
        self.visit = 0

    def setWeight(self, w):
        self.weight = w
    
    def setNodein(self, u):
        self.nodein = u
    
    def setNodeout(self, v):
        self.nodeout = v

    def setVisit(self, n):
        self.visit = n

    def oneVisit(self):
        self.visit += 1
    