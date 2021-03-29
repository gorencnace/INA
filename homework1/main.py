from _collections import deque
import numpy as np
import sys
sys.setrecursionlimit(10000)

files = ['enron']
files2 = ['test']
files3 = ['aps_2010_2011', 'aps_2010_2012', 'aps_2010_2013']


def add_to_stack(i, nodes, s, graph):
    nodes[i] = False
    for j in graph[i]:
        if nodes[j]:
            add_to_stack(j, nodes, s, graph)
    s.append(i)


def invert(graph):
    g = [[] for _ in range(len(graph))]
    for i in range(len(graph)):
        for j in graph[i]:
            g[j].append(i)
    return g


def search(g, i, nodes, comp):
    nodes[i] = False
    comp.add(i)
    for j in g[i]:
        if nodes[j]:
            comp = search(g, j, nodes, comp)
    return comp


def scc(graph):
    nodes = [True for _ in range(len(graph))]
    s = []
    for i in range(len(nodes)):
        if nodes[i]:
            add_to_stack(i, nodes, s, graph)

    gi = invert(graph)
    nodes = [True for _ in range(len(gi))]

    components = []
    while s:
        i = s.pop()
        if nodes[i]:
            components.append(search(gi, i, nodes, set()))
    return components


def distances(graph, i):
    d = [None for _ in range(len(graph))]
    q = deque()
    q.append(i)
    d[i] = 0
    while q:
        i = q.popleft()
        for j in graph[i]:
            if d[j] is None:
                d[j] = d[i] + 1
                q.append(j)
    d = np.array([np.inf if x is None else x for x in d])
    return d


def network_distances(graph):
    d = list()
    for i in range(len(graph)):
        d.append(distances(graph, i))
    return d


# Adjacency list representation
def graph_init():
    for file in ['enron.net']:
        g = []
        m = 0
        with open('data/' + file) as f:
            nodes = 87273
            g = [[] for _ in range(nodes)]
            for line in f:
                if not line.startswith('#'):
                    m += 1
                    a, b = line.split()[:2]
                    node1, node2 = int(a)-1, int(b)-1
                    g[node1].append(node2)
        n = len(g)
        print('========' + file + '========')
        #print('n = ' + str(n))
        #print('m = ' + str(m))
        #print('<k> = ' + str(2 * m / n))
        #print('ro = ' + str(2 * m / (n ** 2 - n)))
        con = scc(g)
        print('# of connected components = ' + str(len(con)))
        print('largest = ' + str(max([len(x) for x in con])))

        '''
        iso = 0
        pendant = 0
        kmax = 0

        for id, node in enumerate(g):
            is_isolated = True
            for e in node:
                if e != id:
                    is_isolated = False
                    break
            if is_isolated or len(node) == 0:
                iso += 1
            elif len(node) == 1:
                pendant += 1
            if kmax < len(node):
                kmax = len(node)

        print('isolated = ' + str(iso))
        print('pendant = ' + str(pendant))
        print('kmax = ' + str(kmax))
        # complexities: n and O(n)
        component_list = components(g)
        print('components = ' + str(len(component_list)))
        print('max component = ' + str(max([len(x) for x in component_list]) / n))

        #lab 2
        d = network_distances(g)
        print('max d = ' + str(max([max(x) for x in d])))
        print('sum d = ' + str(sum([sum(x) for x in d])/(len(g)**2)))
        '''
        #d = np.percentile(np.array(network_distances(g)), 90)
        #print('90 = ' + str(d))



import matplotlib.pyplot as plt
from scipy.stats import poisson
import networkx as nx
from math import ceil

def p_k(g, n):
    tmp = []
    for _, d in g.degree:
        tmp.append(d)
    unique = set(tmp)
    p = [0 for _ in range(max(unique) + 1)]
    for i in unique:
        p[i] = tmp.count(i) / n
    return p

def poisson(k, i):
    return k**i * np.e**(-k) / np.math.factorial(i)

def random_selection(g, t):
    m = g.number_of_edges()
    n = g.number_of_nodes()
    p = [0 for _ in range(n)]
    for i, k in g.degree:
        p[i] = k / (2*m)
    return np.random.choice(n, t, p=p, replace=False)

def complete_graph(n):
    g = nx.Graph()
    for i in range(n):
        g.add_node(i)
        for j in range(0, i):
            g.add_edge(j, i)
    return g

def random_graph(n, k):
    g = complete_graph(ceil(k) + 1)
    for i in range(ceil(k) + 1, n):
        print('Node ' + str(i) + '...')
        print('Progress: ' + str(i/n))
        nodes = random_selection(g, ceil(k/2))
        for j in nodes:
            g.add_edge(i, j)
        print('Complete!')
    return g


def network_init():
    g1 = nx.read_adjlist('data/facebook')
    n = g1.number_of_nodes()
    m = g1.number_of_edges()
    p1 = p_k(g1, n)
    k_avg = 2*m/n
    g2 = nx.gnm_random_graph(n, m)
    p2 = p_k(g2, n)
    p3 = np.array([poisson(k_avg, x) for x in range(7, len(p2))])
    x3 = np.array([x for x in range(7, len(p2))])
    p4 = p_k(random_graph(n, k_avg), n)
    plt.plot(p1, 'bo')
    plt.plot(x3, p3, 'c-')
    plt.plot(p2, 'mo')
    plt.plot(p4, 'rx')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Facebook social network', 'Poisson distribution', 'Erdos-Renyi random graph', 'Random graph'], loc='upper right')
    plt.savefig('plot.png')
    plt.show()

from scipy.stats import spearmanr

def node_position():
    g = [0] * 124
    d = [0] * 124
    c = [0] * 124
    l = [0] * 124
    with open('data/highways') as f:
        for line in f:
            if line.startswith('#'):
                a = line.split()
                i, k = int(a[1])-1, float(a[-1])
                g[i] = k
    graph = nx.read_adjlist('data/highways.net')
    for i, deg in graph.degree:
        ind = int(i)
        d[ind-1] = deg
        c[ind-1] = nx.clustering(graph, i)
        l[ind-1] = nx.closeness_centrality(graph, i)
    print(spearmanr(g, d))
    print(spearmanr(g, l))

    r = list(zip([x for x in range(1,125)], g, l))
    r.sort(key=lambda x: x[2], reverse=True)
    for i in range(10):
        print(r[i])


if __name__ == '__main__':
    # lab
    #graph_init()
    #network_init()
    node_position()
