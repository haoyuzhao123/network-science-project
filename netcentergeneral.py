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
        self.INF = 10000
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
        self.adjlist = []
        for i in range(n):
            self.adjlist.append([])
    
    def addEdge(self, e):
        nout = e.nodeout
        self.adjlist[nout].append(e)


def apsp(g,fn):
    n = g.numnodes
    INF = np.exp(15)
    mat = []
    for i in range(n):
        mat.append([INF] * n)
    for i in range(n):
        for j in range(len(g.adjlist[i])):
            u = g.adjlist[i][j].nodein
            v = g.adjlist[i][j].nodeout
            mat[u][v] = fn(g.adjlist[i][j])
            mat[v][u] = mat[u][v]
            #print(fn)
            #print(mat[u][v])
    for i in range(n):
        mat[i][i] = 0
    for k in range(n):
        for i in range(n):
            for j in range(n):
                mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])

    return mat

def centertree(g, fn):
    INF = np.exp(15)
    res = []
    distmat = apsp(g,fn)
    n = g.numnodes
    maxdist = []
    for i in range(len(distmat)):
        maxdist.append(findMax(distmat[i])[0])
    rootidx = findMin(maxdist)[1]
    mat = []
    for i in range(n):
        mat.append([])
    for i in range(n):
        for j in range(n):
            mat[i].append(INF)
    for i in range(n):
        for j in range(len(g.adjlist[i])):
            u = g.adjlist[i][j].nodein
            v = g.adjlist[i][j].nodeout
            mat[u][v] = fn(g.adjlist[i][j])
            mat[v][u] = mat[u][v]
            #print(mat[u][v])
    isVisit = [0] * n
    isVisit[rootidx] = 1
    queue = []
    for i in range(n):
        if i != rootidx:
            queue.append([mat[rootidx][i], rootidx, i])
    counter = 0
    while True:
        if counter == n-1:
            break
        if len(queue) == 0:
            break
        minidx = 0
        minval = queue[0][0]
        for i in range(1,len(queue)):
            if queue[i][0] < minval:
                minval = queue[i][0]
                minidx = i
        e = queue.pop(minidx)
        if isVisit[e[2]] == 1:
            continue
        else:
            counter += 1
            isVisit[e[2]] = 1
            a = min(e[1],e[2])
            b = max(e[1],e[2])
            res.append(g.adjlist[a][b-a-1])
            for j in range(n):
                if j != e[2]:
                    queue.append([e[0]+mat[e[2]][j],e[2],j])
    #print(len(res))
    return [res,rootidx]


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
        self.INF = 10000
    
    def setup(self,n,t):
        self.numnodes = n
        self.graph.setNumnodes(n)
        self.runningtime = t
        for i in range(n):
            for j in range(i+1,n):
                e = Edge()
                e.setNodes(i,j)
                #set the weights
                e.setWeight(np.random.random() * 4 + 4)
                #e.setWeight(np.random.weibull(0.7) + 4)
                self.graph.addEdge(e)
        a = centertree(self.graph, getweight)
        self.bestedges = a[0]
        self.rootidx = a[1]

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
                #e.lastvalue = np.random.random() * 2 - 1 + e.weight
                e.lastvalue = np.random.random() * 6 - 3 + e.weight
                #e.lastvalue = np.random.weibull(3) * e.weight
                #e.lastvalue = e.weight
                # set the confidence bound
                if e.visit != 0:
                    e.lowerconfidence = e.empmean - 1 * np.sqrt(3 * np.log(self.counter) / (2 * e.visit))
                else:
                    e.lowerconfidence = 0.1

    def postsimu(self):
        self.counter += 1
        ret = centertree(self.graph, getlowerconfidence)
        action = ret[0]
        idx = ret[1]
        actionval = 0
        bestval = 0
        for i in range(len(action)):
            action[i].visit += 1
            #update visit info
            t1 = action[i].visit
            #update action info
            action[i].empmean = (t1-1)/t1 * action[i].empmean + 1 / t1 * action[i].lastvalue
            #actionval += action[i].lastvalue
        #for i in range(len(self.bestedges)):
        #    bestval += self.bestedges[i].lastvalue
        actionval = maxdistfromroot(self.graph,idx,action)
        #bestval = maxdistfromroot(self.graph, self.rootidx, self.bestedges)
        retp = centertree(self.graph, getlastvalue)
        bestaction = retp[0]
        bestidx = retp[1]
        bestval = maxdistfromroot(self.graph,bestidx,bestaction)
        self.actionarmsval.append(actionval)
        self.bestarmsval.append(bestval)
        if (len(self.bestarmsval) % 500 == 0):
            print(actionval)
            print(bestval)
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
            
        for i in range(self.graph.numnodes):
            for j in range(len(self.graph.adjlist[i])):
                print(self.graph.adjlist[i][j].lowerconfidence)
                print(self.graph.adjlist[i][j].visit)
                

def getweight(e):
    return e.weight

def getlowerconfidence(e):
    return e.lowerconfidence

def getlastvalue(e):
    return e.lastvalue

def maxdistfromroot(g,rootidx,action):
    INF = 100000
    n = g.numnodes
    isVisit = [INF] * n
    isVisit[rootidx] = 0
    edges = []
    for i in range(n):
        edges.append([])
    for e in action:
        u = e.nodein
        v = e.nodeout
        e1 = Edge()
        e2 = Edge()
        e1.lastvalue = e.lastvalue
        e2.lastvalue = e.lastvalue
        e1.setNodes(u,v)
        e2.setNodes(v,u)
        edges[u].append(e1)
        edges[v].append(e2)
    dfs(edges,isVisit,rootidx)
    maxidx = 0
    maxval = isVisit[0]
    for i in range(1,n):
        if isVisit[i] > maxval:
            maxval = isVisit[i]
            maxidx = i
    return maxval

def dfs(edges, isVisit,cur):
    INF = 100000
    for e in edges[cur]:
        nextcur = e.nodein
        if isVisit[nextcur] == INF:
            isVisit[nextcur] = e.lastvalue + isVisit[cur]
            dfs(edges,isVisit,nextcur)


def main():
    s = Simulator()
    t = 3000
    s.setup(15,t)
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

def test():
    g = Graph()
    n = 10
    counter = 1
    g.setNumnodes(n)
    for i in range(n):
        for j in range(i+1,n):
            e = Edge()
            e.setNodes(i,j)
            e.setWeight(counter)
            e.lastvalue = counter
            counter += 1
            g.addEdge(e)
    a = centertree(g,getweight)
    action = a[0]
    idx = a[1]
    print(len(action))
    print(maxdistfromroot(g,idx,action))


if __name__ == '__main__':
    main()
    #test()