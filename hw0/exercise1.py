import networkx as nx

f = open('karate_club.adj', 'rb')

G1 = nx.read_adjlist(f)

print('Number of nodes: ' + str(G1.number_of_nodes()))
print('Number of edges: ' + str(G1.number_of_edges()))

f.close()

f = open('facebook_combined.txt', 'rb')

G2 = nx.read_adjlist(f)

print('Number of nodes: ' + str(G2.number_of_nodes()))
print('Number of edges: ' + str(G2.number_of_edges()))
print('Average degree: ' + str(2*G2.number_of_edges()/G2.number_of_nodes()))

f.close()

G3 = nx.gnm_random_graph(G2.number_of_nodes(), G2.number_of_edges())

print('Number of nodes: ' + str(G3.number_of_nodes()))
print('Number of edges: ' + str(G3.number_of_edges()))
print('Average degree: ' + str(2*G3.number_of_edges()/G3.number_of_nodes()))

tmp = nx.pagerank(G2)
pr = sorted(tmp.items(), key=lambda item: item[1], reverse=True)
first10_tmp = pr[:10]
first10 = [(a, b, nx.degree(G2, a)) for a, b in first10_tmp]
print(first10)

tmp = nx.pagerank(G3)
pr = sorted(tmp.items(), key=lambda item: item[1], reverse=True)
first10_tmp = pr[:10]
first10 = [(a, b, nx.degree(G3, a)) for a, b in first10_tmp]
print(first10)


x = 1