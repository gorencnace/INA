import operator
from tqdm import tqdm
import scipy
from cdlib import evaluation, algorithms, classes
import networkx as nx
from random import choices, choice, sample
import community
import igraph
from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
from math import log
from scipy.special import binom
from collections import Counter


class AUC:
    def __init__(self, g):
        self.g = g.copy()
        self.m = self.g.number_of_edges()
        self.n = self.g.number_of_nodes()
        self.communities = {k: v for v in algorithms.louvain(self.g).communities for k in v}

    def preferential_attachment_index(self, i, j):
        return self.g.degree[i] * self.g.degree[j]

    def adamic_adar_index(self, i, j):
        n = nx.common_neighbors(self.g, i, j)
        return sum([1 / log(self.g.degree(k)) for k in n])

    def community_index(self, i, j):
        if self.communities[i] != self.communities[j]:
            return 0
        return self.g.subgraph(self.communities[i]).number_of_edges() / binom(len(self.communities[i]), 2)

    def negative(self, m):
        l = set()
        while len(l) < m:
            a, b = sample(self.g.nodes(), 2)
            if a not in self.g.neighbors(b):
                l.add((a, b))
        return list(l)

    def positive(self, m):
        return sample(self.g.edges(), m)

    def run(self, it):
        size = int(self.m/10)
        l_n = self.negative(size)
        l_p = self.positive(size)
        self.g.remove_edges_from(l_p)

        pa_m1, pa_m2 = 0, 0
        aa_m1, aa_m2 = 0, 0
        com_m1, com_m2 = 0, 0

        with tqdm(total=size, desc='Iteracija: ' + str(it), unit='calc') as prog_bar:
            for _ in range(size):
                n = choice(l_n)
                p = choice(l_p)

                if self.preferential_attachment_index(*n) < self.preferential_attachment_index(*p):
                    pa_m1 += 1
                else:
                    pa_m2 += 1

                if self.adamic_adar_index(*n) < self.adamic_adar_index(*p):
                    aa_m1 += 1
                else:
                    aa_m2 += 1

                if self.community_index(*n) < self.community_index(*p):
                    com_m1 += 1
                else:
                    com_m2 += 1

                prog_bar.update(1)

        return (pa_m1 + pa_m2 / 2) / size, (aa_m1 + aa_m2 / 2) / size, (com_m1 + com_m2 / 2) / size


def girvan_newman_graph(mixing, num_groups=3, num_nodes=24, exp_degree=20):
    g = nx.Graph()
    g.add_nodes_from([x for x in range(num_nodes * num_groups)])


    within_group_prob = (1 - mixing) * exp_degree / num_nodes
    between_groups_prob = mixing * exp_degree / (num_nodes * (num_groups - 1))

    edges = []
    for i in g.nodes():
        for j in g.nodes():
            if i < j:
                if i % num_groups == j % num_groups:
                    if choices([True, False], [within_group_prob, 1 - within_group_prob])[0]:
                        edges.append((i, j))
                else:
                    if choices([True, False], [between_groups_prob, 1 - between_groups_prob])[0]:
                        edges.append((i, j))
    g.add_edges_from(edges)
    return g
'''
def gn_benchmark():
    l = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    l_a = []
    l_b = []
    l_c = []
    for mi in l:
        a = 0
        b = 0
        c = 0
        for _ in range(25):
            g = girvan_newman_graph(mi)
            louvain = [None] * 72
            infomap = [None] * 72
            label_prop = [None] * 72
            true_labels = [i % 3 for i in range(72)]

            tmp = community.best_partition(g)  # https://perso.crans.org/aynaud/communities/
            for k, v in tmp.items():
                louvain[k] = v
            # part2 = list(nx.algorithms.community.label_propagation_communities(g)) # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.label_propagation.label_propagation_communities.html#networkx.algorithms.community.label_propagation.label_propagation_communities
            g2 = igraph.Graph()
            g2.add_vertices(g.nodes())
            g2.add_edges(g.edges())
            tmp = list(g2.community_infomap())
            for v in range(len(tmp)):
                for k in tmp[v]:
                    infomap[k] = v
            tmp = list(g2.community_label_propagation())
            for v in range(len(tmp)):
                for k in tmp[v]:
                    label_prop[k] = v

            a += normalized_mutual_info_score(true_labels, louvain)
            b += normalized_mutual_info_score(true_labels, infomap)
            c += normalized_mutual_info_score(true_labels, label_prop)
        l_a.append(a/25)
        l_b.append(b/25)
        l_c.append(c/25)
    plt.plot(l, l_a, 'ro')
    plt.plot(l, l_b, 'bo')
    plt.plot(l, l_c, 'go')
    plt.ylabel('NMI')
    plt.xlabel('mi')
    plt.legend(['louvain', 'infomap', 'label propagation'], loc='lower left')
    plt.savefig('plot31.png')
    plt.show()
'''
def gn_benchmark():
    l = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
    l_a = []
    l_b = []
    l_c = []
    for mi in l:
        a = 0
        b = 0
        c = 0
        for _ in range(25):
            g = girvan_newman_graph(mi)
            louvain = algorithms.louvain(g)
            walktrap = algorithms.walktrap(g)
            label_prop = algorithms.label_propagation(g)
            true_labels = classes.NodeClustering([[3*i + j for i in range(24)] for j in range(3)], g)

            a += evaluation.normalized_mutual_information(true_labels, louvain).score
            b += evaluation.normalized_mutual_information(true_labels, walktrap).score
            c += evaluation.normalized_mutual_information(true_labels, label_prop).score
        l_a.append(a/25)
        l_b.append(b/25)
        l_c.append(c/25)
    plt.plot(l, l_a, 'ro')
    plt.plot(l, l_b, 'bo')
    plt.plot(l, l_c, 'go')
    plt.ylabel('NMI')
    plt.xlabel('mi')
    plt.legend(['louvain', 'walktrap', 'label propagation'], loc='lower left')
    plt.savefig('plot31.png')
    plt.show()

def gt(file):
    f = open(file + '.net')
    f.readline()
    sez = dict()#{k: None for k in range(90162)}#62#2500
    for line in f.readlines():
        s = line.split()
        if s[0] == '*arcs':
            break
        sez[s[1][1:-1]] = int(s[2])
    return sez

def create_graph(file):
    f = open(file + '.net')
    g = nx.Graph()
    g.add_nodes_from([x for x in range(62)]) #2500
    while not f.readline().startswith('*edges'):
        pass
    for line in f.readlines():
        s = line.split()
        g.add_edge(int(s[0])-1, int(s[1])-1)
    return g


def lan_benchmark():
    lab = ['00', '02', '04', '06', '08']
    l_a = []
    l_b = []
    l_c = []
    for l in lab:
        a = 0
        b = 0
        c = 0
        for i in range(25):
            name = 'data/LFR/LFR_' + l + '_' + str(i)
            g = create_graph(name)
            louvain = algorithms.louvain(g)
            walktrap = algorithms.walktrap(g)
            label_prop = algorithms.label_propagation(g)
            true_labels = classes.NodeClustering(gt(name), g)

            a += evaluation.normalized_mutual_information(true_labels, louvain).score
            b += evaluation.normalized_mutual_information(true_labels, walktrap).score
            c += evaluation.normalized_mutual_information(true_labels, label_prop).score
        l_a.append(a / 25)
        l_b.append(b / 25)
        l_c.append(c / 25)
    l = [0, 0.2, 0.4, 0.6, 0.8]
    plt.plot(l, l_a, 'ro')
    plt.plot(l, l_b, 'bo')
    plt.plot(l, l_c, 'go')
    plt.ylabel('NMI')
    plt.xlabel('mi')
    plt.legend(['louvain', 'walktrap', 'label propagation'], loc='lower left')
    plt.savefig('plot32.png')
    plt.show()


def er_benchmark():
    l = [8, 16, 24, 32, 40]
    l_a = []
    l_b = []
    l_c = []
    truth = [[i for i in range(1000)]]
    for k in l:
        a = 0
        b = 0
        c = 0
        for _ in range(25):
            g = nx.gnm_random_graph(1000, 1000*k)
            true_labels = classes.NodeClustering(truth, g)
            louvain = algorithms.louvain(g)
            walktrap = algorithms.walktrap(g)
            label_prop = algorithms.label_propagation(g)

            a += evaluation.variation_of_information(true_labels, louvain).score
            b += evaluation.variation_of_information(true_labels, walktrap).score
            c += evaluation.variation_of_information(true_labels, label_prop).score
        l_a.append(a / (25*log(1000)))
        l_b.append(b / (25*log(1000)))
        l_c.append(c / (25*log(1000)))

    plt.plot(l, l_a, 'ro')
    plt.plot(l, l_b, 'bo')
    plt.plot(l, l_c, 'go')
    plt.ylabel('NVI')
    plt.xlabel('k')
    plt.legend(['louvain', 'walktrap', 'label propagation'], loc='center right')
    plt.savefig('plot33b.png')
    plt.show()


def dolphins_benchmark():
    g = create_graph('data/dolphins')
    louvain = []
    walktrap = []
    label_prop = []
    for i in range(25):
        louvain.append(algorithms.louvain(g, randomize=i))
        walktrap.append(algorithms.walktrap(g))
        label_prop.append(algorithms.label_propagation(g))
    a, b, c = [], [], []
    for i in range(25):
        for j in range(i+1, 25):
            a.append(evaluation.variation_of_information(louvain[i], louvain[j]).score)
            b.append(evaluation.variation_of_information(walktrap[i], walktrap[j]).score)
            c.append(evaluation.variation_of_information(label_prop[i], label_prop[j]).score)

    print(sum(a) / (len(a)*log(62)))
    print(sum(b) / (len(b)*log(62)))
    print(sum(c) / (len(c)*log(62)))


def auc_z():
    times = 10
    er_g = nx.gnm_random_graph(25000, 250000)
    p2p = nx.read_adjlist('data/gnutella')
    fb = nx.read_adjlist('data/circles')
    nec = nx.read_adjlist('data/nec')

    p, a, c = 0, 0, 0
    for i in range(times):
        auc = AUC(er_g)
        r = auc.run(i)
        p += r[0]
        a += r[1]
        c += r[2]
    print(str(p/times) + ', ' + str(a/times) + ', ' + str(c/times))

    p, a, c = 0, 0, 0
    for i in range(times):
        auc = AUC(p2p)
        r = auc.run(i)
        p += r[0]
        a += r[1]
        c += r[2]
    print(str(p/times) + ', ' + str(a/times) + ', ' + str(c/times))

    p, a, c = 0, 0, 0
    for i in range(times):
        auc = AUC(fb)
        r = auc.run(i)
        p += r[0]
        a += r[1]
        c += r[2]
    print(str(p/times) + ', ' + str(a/times) + ', ' + str(c/times))

    p, a, c = 0, 0, 0
    for i in range(times):
        auc = AUC(nec)
        r = auc.run(i)
        p += r[0]
        a += r[1]
        c += r[2]
    print(str(p/times) + ', ' + str(a/times) + ', ' + str(c/times))

def citation():
    g = nx.read_pajek('data/aps_2008_2013.net')
    truth = gt('data/aps_2008_2013')
    t = 0
    all = 0
    for _ in range(10):
        walktrap = algorithms.walktrap(g).communities
        d = {k: [] for k in range(len(walktrap))}
        for k, coms in enumerate(walktrap):
            for art in coms:
                if "2013" not in art:
                    d[k].append(truth[art])
        for k, coms in enumerate(walktrap):
            if len(d[k]) != 0:
                com, _ = max(Counter(d[k]).items(), key=lambda x:x[1])
                for a in coms:
                    if "2013" in a:
                        if truth[a] == com:
                            t += 1
                        all += 1
            else:
                all += len(coms)

    print(t/all)


if __name__ == '__main__':
    #gn_benchmark()
    #lan_benchmark()
    #er_benchmark()
    #dolphins_benchmark()
    #auc_z()
    citation()
    x=1
