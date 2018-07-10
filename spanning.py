import numpy as np
import scipy as scp
import matplotlib.pyplot as plt

class Edge:
    def __init__(self):
        self.INF = 100
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
    

class Unionfind:
    def __init__(self, n):
        self.par = [0] * n
        self.depth = [1] * n
        for i in range(n):
            self.par[i] = i
    
    def find(self, u):
        if self.par[u] == u:
            return u
        else:
            self.par[u] = self.find(self.par[u])
            return self.par[u]
    
    def union(self, u, v):
        paru = self.find(u)
        parv = self.find(v)
        if self.depth[paru] > self.depth[parv]:
            self.par[parv] = paru
        else:
            self.par[paru] = parv
            if self.depth[paru] == self.depth[parv]:
                self.depth[parv] += 1

class Graph:
    def __init__(self):
        self.adjlist = []
        self.numnodes = 0

    def setNumnodes(self, n):
        self.numnodes = n
        self.adjlist = []
        for i in range(n):
            self.adjlist.append([])
    
    def addEdge(self, e):
        nout = e.nodeout
        #print(nout)
        self.adjlist[nout].append(e)

def getweight(e):
    return e.weight

def getlowerconfidence(e):
    return e.lowerconfidence

def kruskal(g, fn):
    res = []
    n = g.numnodes
    uf = Unionfind(n)
    count = 0
    edges = []
    for i in range(n):
        edges.extend(g.adjlist[i])
    edges.sort(key = fn)
    for i in range(len(edges)):
        e = edges[i]
        if uf.find(e.nodein) != uf.find(e.nodeout):
            res.append(e)
            count += 1
            uf.union(e.nodein,e.nodeout)
            if count == n-1:
                break
    return res

class Simulator:
    def __init__(self):
        self.graph = Graph()
        self.numnodes =0
        self.runningtime = 0
        self.bestedges = []
        self.counter = 0
        self.bestarmsval = []
        self.actionarmsval = []
        self.cumbestarmsval = []
        self.cumactionarmsval = []
    
    def setup(self,n,t):
        self.numnodes = n
        self.graph.setNumnodes(n)
        self.runningtime = t
        for i in range(n):
            for j in range(i+1,n):
                e = Edge()
                e.setNodes(i,j)
                #set the weights
                #e.setWeight(np.random.random() * 2 + 3)
                e.setWeight(np.random.weibull(0.8))
                self.graph.addEdge(e)
                #for i in range(self.numnodes):
                #    print(len(self.graph.adjlist[i]))
        self.bestedges = kruskal(self.graph, getweight)
        #for i in range(self.numnodes):
            #print(len(self.graph.adjlist[i]))

    def presimu(self):
        for i in range(self.numnodes):
            for j in range(len(self.graph.adjlist[i])):
                e = self.graph.adjlist[i][j]
                # set the value of the random variable
                """
                temp = np.random.random()
                
                if temp > e.weight:
                    e.lastvalue = 0
                else:
                    e.lastvalue = 1
                    """
                #e.lastvalue = np.random.random() * 20 - 10 + e.weight
                #e.lastvalue = np.random.random() * 6 - 3 + e.weight
                e.lastvalue = np.random.weibull(0.8) * e.weight
                #e.lastvalue = e.weight
                # set the confidence bound
                if e.visit != 0:
                    e.lowerconfidence = e.empmean - e.empmean * np.sqrt(3 * np.log(self.counter) / (2 * e.visit))

    def postsimu(self):
        self.counter += 1
        action = kruskal(self.graph, getlowerconfidence)
        actionval = 0
        bestval = 0
        for i in range(len(action)):
            action[i].visit += 1
            #update visit info
            t1 = action[i].visit
            #update action info
            action[i].empmean = (t1-1)/t1 * action[i].empmean + 1 / t1 * action[i].lastvalue
            actionval += action[i].lastvalue
        for i in range(len(self.bestedges)):
            bestval += self.bestedges[i].lastvalue
        self.actionarmsval.append(actionval)
        self.bestarmsval.append(bestval)
        if len(self.cumactionarmsval) == 0:
            self.cumactionarmsval.append(actionval)
            self.cumbestarmsval.append(bestval)
        else:
            self.cumactionarmsval.append(self.cumactionarmsval[-1] + actionval)
            self.cumbestarmsval.append(self.cumbestarmsval[-1] +bestval)

    def wholesimu(self):
        for i in range(self.runningtime):
            self.presimu()
            self.postsimu()
        
        for i in range(self.numnodes):
            for j in range(len(self.graph.adjlist[i])):
                print(self.graph.adjlist[i][j].visit)
                



def main():
    s = Simulator()
    t = 3000
    s.setup(20,t)
    """
    print(s.graph.numnodes)
    print(len(s.bestedges))
    for i in range(len(s.bestedges)):
        print(s.bestedges[i].weight)
        """
    s.wholesimu()
    cumact = np.array(s.cumactionarmsval)
    cumvst = np.array(s.cumbestarmsval)
    plt.plot(cumact)
    plt.plot(cumvst)
    plt.plot(cumact - cumvst)
    plt.legend(['Cumulative Cost','Cumulative Best Action Cost',\
            'Regret'],loc='upper left')
    plt.xlabel('Rounds')
    plt.ylabel('Costs')
    plt.title('Costs and Regret')
    plt.show()
    temp = list(range(t))
    coor = np.array(temp) + 1
    plt.plot((cumact - cumvst) / coor)
    plt.legend('Mean Regret')
    plt.title('Mean Regret')
    plt.xlabel('Rounds')
    plt.ylabel('Cost')
    plt.show()

if __name__ == '__main__':
    main()
    