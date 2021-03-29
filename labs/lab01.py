from _collections import deque

files = ['toy', 'karate_club', 'collaboration_imdb', 'www_google']
files2 = ['toy']

# Number and size of connected components
def weak_component(graph, nodes, node):
    connected = []
    stack = [node]
    nodes.remove(node)
    while len(stack) > 0:
        node = stack.pop()
        connected.append(node)
        for j in graph[node]:
            if j in nodes:
                stack.append(j)
                nodes.remove(j)
    return connected


def components(graph):
    nodes = set(range(len(graph)))
    connected_lists = []
    while len(nodes) > 0:
        connected_lists.append(weak_component(graph, nodes, next(iter(nodes))))
    return connected_lists


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
    d = [0 if x is None else x for x in d]
    return d


def network_distances(graph):
    d = list()
    for i in range(len(graph)):
        d.append(distances(graph, i))
    return d


# Adjacency list representation
def graph_init():
    for file in files2:
        g = []
        m = 0
        with open('data/' + file + '.net') as f:
            for line in f:
                if line.startswith('*vertices'):
                    nodes = int(line.split()[1])
                    g = [[] for _ in range(nodes)]
                elif line.startswith('*'):
                    break
            for line in f:
                m += 1
                a, b = line.split()[:2]
                node1, node2 = int(a) - 1, int(b) - 1
                g[node1].append(node2)
                g[node2].append(node1)
        n = len(g)
        print('========' + file + '========')
        print('n = ' + str(n))
        print('m = ' + str(m))
        print('<k> = ' + str(2 * m / n))
        print('ro = ' + str(2 * m / (n ** 2 - n)))

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


# N - mnozica vozlisc, katerih se nismo pregledali (mnozica, da je remove hitrejsi)
# S - stack, na katerem so vozlisca, ki smo jih ze videli, nismo pa se pogledali njegovih sosedov
# C - seznam, v katerem so vozlisca iz stacka, preden pogledamo njegove sosede, ga damo notr
# Gamma_i - sosescina vozlisca i
# O(m)
# preiskovanje v globino, ker dajemo na zacetek stacka (nizja prostorska kompleksnost, ker ne hranimo celotne sosescine)


# TODO: How could you further improve the algorithm to only compute s and S?
# s - stevec na 0, ko najde novo vozlisce ga damo na +1 - bolj ucinkovito pri stevilnih majhnjih komponentah (miljon vozlisc z malimi komponentami, miljon seznamov - veliko prostora - zelo pocasi)
#

# Average node distance and network diameter
# D.. seznam, ki nam pove, kaj je razdalja od vozlisca i do vseh ostalih (i-i je 0)
# na vsakem koraku, ko najdemo nove sosede, moramo nastat razdalje, če jih še nismo videli
# če je razdalja nedefinirana, smo najdl novo vozlisce
# izracunamo razdaljo tako, da pogledamo njegove sosede, razdalja i-j je radalja do j + 1
# - Preiskovanje v sirino (queue), da pridemo do potimalne resitve
# - kompleksnost: desno: O(m), celotno: O(n*m)
# ce imamo usmerjen graf, gledamo samo vse naslednike

# TODO: How is the algorithm different from the famous Dijkstra’s algorithm? In which case you would necessarily have to use the Dijkstra’s algorithm?
# nujno bi moral uporabiti, ce bi imeli na povezavi utezi
# ce imamo na povezavi utezi, treba gledati najprej najkrajse poti - uporaba kopice (v korenu najmanjse elemente)

# TODO:  How could you speed up the algorithm to only approximate ⟨d⟩ and dmax?
# Namesto da delamo bf search na vsakem vozliscu, delamo na 100 nakljucnih vozliscih, dobili bomo dobro idejo orazdaljah v omrezju (od O(nm) do O(m)

#46:30
# TODO: Average node clustering coefficient
# podatktura za graf: dictionary, kaj podobnega (matrika je prevec prostorsko kompleksna) - moramo znati ucinkovitu prevert povezave (ce sta dve vozlisci povezani)
# Bolj se splaca stet trikotnike cez povezave kot cez vozlisca:
# za neko povezavo nas zanima, koliko trikotniko tece cez. Gremo cez use povezave, ki s drzijo itega vozlsca in za vsako povezavo izracunamo, v koliko trikotnikih je
# na povezavi pogledamo, katero vozlisce ima manjso stopnjo, gledamo cez tisto vozlisce
# O(n * <k**2>) vs O(n * <k>**2) (kvadrat povprecja vs povprecje kvadratov)
# rezultati: 1:07:30
# napaka pri average, c se ne overrida ampak sesteje (1:08:00)

# TODO: Erdos-Renyi
# razlika: desn lahko vsebuje vzporedne povezave in cikle


if __name__ == '__main__':
    # lab
    graph_init()
