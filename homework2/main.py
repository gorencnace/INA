import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from _collections import deque


def e1():
    dolphins = nx.read_adjlist('data/dolphins')
    bc = nx.betweenness_centrality(dolphins, normalized=True)
    print('Betweenness centrality:')
    print({key: value for key, value in sorted(bc.items(), key=lambda item: item[1], reverse=True)[:5]})

    ev = nx.eigenvector_centrality(dolphins)
    print('Eigenvector centrality:')
    print({key: value for key, value in sorted(ev.items(), key=lambda item: item[1], reverse=True)[:5]})

    kc = nx.katz_centrality(dolphins)
    print('Katz centrality:')
    print({key: value for key, value in sorted(kc.items(), key=lambda item: item[1], reverse=True)[:5]})

    pr = nx.pagerank(dolphins)
    print('PageRank centrality:')
    print({key: value for key, value in sorted(pr.items(), key=lambda item: item[1], reverse=True)[:5]})


def read_undirected(path):
    g = nx.DiGraph()
    with open(path) as file:
        for line in file:
            if not line.startswith('#'):
                a = int(line.split()[0])
                b = int(line.split()[1])
                g.add_edge(a, b)
    return g


def generate_plot(s):
    g = read_undirected('data/' + s)
    tmp = [x for _, x in g.degree()]
    p_k = [tmp.count(i) / g.number_of_nodes() for i in range(max(tmp) + 1)]
    plt.plot(p_k, 'bo')
    tmp = [x for _, x in g.in_degree()]
    p_k = [tmp.count(i) / g.number_of_nodes() for i in range(max(tmp) + 1)]
    plt.plot(p_k, 'ro')
    tmp = [x for _, x in g.out_degree()]
    p_k = [tmp.count(i) / g.number_of_nodes() for i in range(max(tmp) + 1)]
    plt.plot(p_k, 'go')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Degree distribution', 'In-degree distribution', 'Out-degree distribution'],
               loc='upper right')
    plt.xlabel('k')
    plt.ylabel('p_k')
    plt.savefig(s + '.png')
    plt.show()


def gamma(s):
    k = [i for i in range(1, 101)]
    gamma = []
    g = read_undirected('data/' + s)
    tmp = [x for _, x in g.in_degree()]
    for k_min in k:
        n = sum([i >= k_min for i in tmp])
        gamma.append(1 + n / (sum([np.log(i /(k_min - 0.5)) if i >= k_min else 0 for i in tmp])))
    plt.plot(k, gamma)
    plt.xlabel('k_min')
    plt.ylabel('gamma')
    plt.savefig(s + '_gamma.png')
    plt.show()
    n = sum([i >= 25 for i in tmp])
    print(1 + n / (sum([np.log(i / (25 - 0.5)) if i >= 25 else 0 for i in tmp])))


def e2():
    software = ['java', 'lucene']
    for s in software:
        generate_plot(s)
        gamma(s)

def node_removal_list(n, p):
    l = []
    for i in n:
        if np.random.choice(np.arange(0,2), p=[1-p, p]):
            l.append(i)
    return l


def e3():
    fractions = [0, 0.1, 0.1, 0.1, 0.1, 0.1]
    f = [0, .1, .2, .3, .4, .5]
    pl = []
    g = nx.read_adjlist('data/nec')
    for p in fractions:
        rem = node_removal_list(list(g.nodes), p)
        g.remove_nodes_from(rem)
        largest_cc = max(nx.connected_components(g), key=len)
        pl.append(len(largest_cc) / g.number_of_nodes())
    plt.plot(f, pl)
    pl = []
    g = nx.read_adjlist('data/nec')
    for p in fractions:
        rem = int(p*g.number_of_nodes())
        l = list(g.degree())
        l.sort(key=lambda x: x[1], reverse=True)
        tmp = [x for x, _ in l[:rem]]
        g.remove_nodes_from(tmp)
        largest_cc = max(nx.connected_components(g), key=len)
        pl.append(len(largest_cc) / g.number_of_nodes())
    plt.plot(f, pl)
    plt.legend(['Errors', 'Attacks'], loc='lower left')
    plt.xlabel('fraction of removed nodes')
    plt.ylabel('fraction of nodes in the largest connected component')
    plt.savefig('internet.png')
    plt.show()

    pl = []
    g = nx.read_adjlist('data/nec')
    n = g.number_of_nodes()
    m = g.number_of_edges()
    g = nx.gnm_random_graph(n, m)
    for p in fractions:
        rem = node_removal_list(list(g.nodes), p)
        g.remove_nodes_from(rem)
        largest_cc = max(nx.connected_components(g), key=len)
        pl.append(len(largest_cc) / g.number_of_nodes())
    plt.plot(f, pl)
    pl = []
    g = nx.gnm_random_graph(n, m)
    for p in fractions:
        rem = int(p * g.number_of_nodes())
        l = list(g.degree())
        l.sort(key=lambda x: x[1], reverse=True)
        tmp = [x for x, _ in l[:rem]]
        if tmp:
            g.remove_nodes_from(tmp)
        largest_cc = max(nx.connected_components(g), key=len)
        pl.append(len(largest_cc) / g.number_of_nodes())
    plt.plot(f, pl)
    plt.legend(['Errors', 'Attacks'], loc='lower left')
    plt.xlabel('fraction of removed nodes')
    plt.ylabel('fraction of nodes in the largest connected component')
    plt.savefig('er.png')
    plt.show()


def distances(graph, i):
    d = {i: np.infty for i in list(graph.nodes)}
    q = deque()
    q.append(i)
    d[i] = 0
    while q:
        i = q.popleft()
        for j in graph[i]:
            if d[j] == np.infty:
                d[j] = d[i] + 1
                q.append(j)
    return d


def average_distance(g):
    d = list()
    for i in list(g.nodes):
        d.append(distances(g, i))
    n = g.number_of_nodes()
    return sum([v for tmp in d for v in tmp.values()]) / (n * (n-1))


def e4():
    g = nx.read_adjlist('data/social')
    rand_node = list(g.nodes)[np.random.randint(g.number_of_nodes())]

    walk = {rand_node}
    while True:
        if len(walk) * 10 >= g.number_of_nodes():
            break
        adj = list(g.adj[rand_node])
        rand_node = adj[np.random.randint(len(adj))]
        walk.add(rand_node)
    g_ind = g.subgraph(walk)
    print('AIDS <C> = ' + str(nx.average_clustering(g_ind)))
    print('AIDS <d> = ' + str(average_distance(g_ind)))
    print('AIDS : ' + str(np.log(g_ind.number_of_nodes()) / np.log(sum([x for _, x in list(g_ind.degree())]) / g_ind.number_of_nodes())))
    print('Social <C> = ' + str(nx.average_clustering(g)))
    print('Social <d> = ' + str(average_distance(g)))
    print('S : ' + str(np.log(g.number_of_nodes()) / np.log(sum([x for _, x in list(g.degree())]) / g.number_of_nodes())))

    tmp = [x for _, x in g.degree()]
    p_k = [tmp.count(i) / g.number_of_nodes() for i in range(max(tmp) + 1)]
    plt.plot(p_k, 'bo')
    tmp = [x for _, x in g_ind.degree()]
    p_k = [tmp.count(i) / g_ind.number_of_nodes() for i in range(max(tmp) + 1)]
    plt.plot(p_k, 'ro')
    plt.yscale('log')
    plt.xscale('log')
    plt.legend(['Original social network', 'Sampled social network', 'Out-degree distribution'],
               loc='upper right')
    plt.xlabel('k')
    plt.ylabel('p_k')
    plt.savefig('aids.png')
    plt.show()


if __name__ == '__main__':
    e2()