files = ['toy', 'karate_club', 'www_google']


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

# Adjacency list representation
def graph_init():
    for file in files:
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
        print('<k> = ' + str(2*m/n))
        print('ro = ' + str(2*m/(n**2 - n)))

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

# N - mnozica vozlisc, katerih se nismo pregledali (mnozica, da je remove hitrejsi)
# S - stack, na katerem so vozlisca, ki smo jih ze videli, nismo pa se pogledali njegovih sosedov
# C - seznam, v katerem so vozlisca iz stacka, preden pogledamo njegove sosede, ga damo notr
# Gamma_i - sosescina vozlisca i
# O(m)
# preiskovanje v globino, ker dajemo na zacetek stacka (nizja prostorska kompleksnost, ker ne hranimo celotne sosescine)


if __name__ == '__main__':
    graph_init()
