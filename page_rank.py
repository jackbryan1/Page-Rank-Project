import os
import time
from networkx import DiGraph
from progress import Progress
import random

WEB_DATA = os.path.join(os.path.dirname(__file__), 'school_web.txt')


def load_graph(fd):
    """Load graph from text file

    Parameters:
    fd -- a file like object that contains lines of URL pairs

    Returns:
    A dict mapping a URL (str) to a list of target URLs (str).
    """
    # Iterate through the file line by line
    graph=DiGraph()
    for line in fd:
        # And split each line into two URLs
        node, target = line.split()
        if node not in graph:
            graph.add_node(node) #Adds a node only if it has not already been found

        graph.add_edge(node, target) #All edges are unique so are always added

    return graph

def print_stats(graph):
    """Print number of nodes and edges in the given graph"""
    print(f"The number of nodes is: {graph.number_of_edges()}")
    print(f"The number of edges is: {graph.number_of_nodes()}") #These statements print out the number of nodes and edges in the graph

def stochastic_page_rank(graph, n_iter=1_000_000, n_steps=100):
    """Stochastic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of random walks performed
    n_steps (int) -- number of followed links before random walk is stopped

    Returns:
    A dict that assigns each page its hit frequency

    This function estimates the Page Rank by counting how frequently
    a random walk that starts on a random node will after n_steps end
    on each node of the given graph.
    """
    hitcount = dict()
    for node in graph:
        hitcount[node] = 0 #Initialises the hitcount

    for i in range(n_iter):
        currentnode = random.choice(list(graph.nodes)) #Chooses a random for the walker to start at
        for j in range(n_steps):
            outedge = random.choice(list(graph.out_edges(currentnode))) #Chooses a random outedge
            currentnode = outedge[1] #Gets the sink of the outedge
        if i == 0:
            hitcount[currentnode] += 1 #Prevents a division by 0
        else:
            hitcount[currentnode] += 1/i #Adds to the hitcount
    return hitcount

def distribution_page_rank(graph, n_iter=100):
    """Probabilistic PageRank estimation

    Parameters:
    graph -- a graph object as returned by load_graph()
    n_iter (int) -- number of probability distribution updates

    Returns:
    A dict that assigns each page its probability to be reached

    This function estimates the Page Rank by iteratively calculating
    the probability that a random walker is currently on any node.
    """
    nodeprob = dict()
    for node in graph:
        nodeprob[node] = 1/(graph.number_of_nodes()) #Initialises the probability of a node to be evenly spread

    for i in range(n_iter):
        nextprob = dict()
        for node in graph:
            nextprob[node] = 0 #Initialises the probability of the next node
        for node in graph:
            p = nodeprob[node]/(graph.out_degree(node)) #Determines the probability
            for j in graph.out_edges(node):
                nextprob[j[1]] += p #Adds the probability
        nodeprob = nextprob #Copies over the values
    return nodeprob

def main():
    # Load the web structure from file
    web = load_graph(open(WEB_DATA))

    # print information about the website
    print_stats(web)

    # The graph diameter is the length of the longest shortest path
    # between any two nodes. The number of random steps of walkers
    # should be a small multiple of the graph diameter.
    diameter = 3

    # Measure how long it takes to estimate PageRank through random walks
    print("Estimate PageRank through random walks:")
    n_iter = len(web)**2
    n_steps = 2*diameter
    start = time.time()
    ranking = stochastic_page_rank(web, n_iter, n_steps)
    stop = time.time()
    time_stochastic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
    print(f'Calculation took {time_stochastic:.2f} seconds.\n')

    # Measure how long it takes to estimate PageRank through probabilities
    print("Estimate PageRank through probability distributions:")
    n_iter = 2*diameter
    start = time.time()
    ranking = distribution_page_rank(web, n_iter)
    stop = time.time()
    time_probabilistic = stop - start

    # Show top 20 pages with their page rank and time it took to compute
    top = sorted(ranking.items(), key=lambda item: item[1], reverse=True)
    print('\n'.join(f'{100*v:.2f}\t{k}' for k,v in top[:20]))
    print(f'Calculation took {time_probabilistic:.2f} seconds.\n')

    # Compare the compute time of the two methods
    speedup = time_stochastic/time_probabilistic
    print(f'The probabilitic method was {speedup:.0f} times faster.')


if __name__ == '__main__':
    main()
