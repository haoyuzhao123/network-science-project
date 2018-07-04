class Edge:
    def __init__(self):
        self.INF = -100
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
        self.adjlist = [[]] * n
    
    def addEdge(self, e):
        nout = e.nodeout
        self.adjlist[nout].append(e)

def getweight(e):
    return e.weight

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

def main():
    g = Graph()
    g.setNumnodes(5)
    e0 = Edge()
    e1 = Edge()
    e2 = Edge()
    e3 = Edge()
    e4 = Edge()
    e5 = Edge()
    e0.setWeight(0)
    e1.setWeight(1)
    e2.setWeight(2)
    e3.setWeight(3)
    e4.setWeight(4)
    e5.setWeight(5)
    e0.setNodes(0,1)
    e1.setNodes(2,4)
    e2.setNodes(1,4)
    e3.setNodes(1,2)
    e4.setNodes(3,4)
    e5.setNodes(2,3)
    g.addEdge(e0)
    g.addEdge(e1)
    g.addEdge(e2)
    g.addEdge(e3)
    g.addEdge(e4)
    g.addEdge(e5)
    res = kruskal(g,getweight)
    e0.setWeight(10)
    for i in range(len(res)):
        print(res[i].weight)


if __name__ == '__main__':
    main()
    