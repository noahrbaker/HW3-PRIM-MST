import numpy as np
import heapq
from typing import Union

class Graph:

    def __init__(self, adjacency_mat: Union[np.ndarray, str]):
        """
    
        Unlike the BFS assignment, this Graph class takes an adjacency matrix as input. `adjacency_mat` 
        can either be a 2D numpy array of floats or a path to a CSV file containing a 2D numpy array of floats.

        In this project, we will assume `adjacency_mat` corresponds to the adjacency matrix of an undirected graph.
    
        """
        if type(adjacency_mat) == str:
            self.adj_mat = self._load_adjacency_matrix_from_csv(adjacency_mat)
        elif type(adjacency_mat) == np.ndarray:
            self.adj_mat = adjacency_mat
        else: 
            raise TypeError('Input must be a valid path or an adjacency matrix')
        
        # added to account for empty inputs
        if self.adj_mat.size == 0:
            raise ValueError('adjacency matrix input cannot be empty')
        
        self.mst = None
        self.tups = None

    def _load_adjacency_matrix_from_csv(self, path: str) -> np.ndarray:
        with open(path) as f:
            return np.loadtxt(f, delimiter=',')

    def construct_mst(self):
        """
    
        TODO: Given `self.adj_mat`, the adjacency matrix of a connected undirected graph, implement Prim's 
        algorithm to construct an adjacency matrix encoding the minimum spanning tree of `self.adj_mat`. 
            
        `self.adj_mat` is a 2D numpy array of floats. Note that because we assume our input graph is
        undirected, `self.adj_mat` is symmetric. Row i and column j represents the edge weight between
        vertex i and vertex j. An edge weight of zero indicates that no edge exists. 
        
        This function does not return anything. Instead, store the adjacency matrix representation
        of the minimum spanning tree of `self.adj_mat` in `self.mst`. We highly encourage the
        use of priority queues in your implementation. Refer to the heapq module, particularly the 
        `heapify`, `heappop`, and `heappush` functions.

        """
        # list of the best tuples containing weight, current node, and previous node
        best_tups = []

        num_nodes = self.adj_mat.shape[0]
        
        # list of whether or not the node has been visited
        # because of symmetry, this can be applied along the rows and cols
        visited = [False] * num_nodes
        
        # heap, containing node information to pass through
        # (weight, current node, previous node)
        heap_q = [(0, 0, -1)]

        # this will continue until N-1 nodes
        while len(best_tups) < num_nodes - 1:
            weight, node, prev = heapq.heappop(heap_q)
            
            # if the node has been visited, move to the next heap_q
            if visited[node]:
                continue

            # mark it as visited because we are here now
            visited[node] = True

            # doesn't add anything if it is our first time around
            # otherwise it will append to our best_tups list
            if prev != -1:
                best_tups.append((weight, prev, node))

            # for all the connected nodes 
            for neighbor in range(num_nodes):
                # 0 evals as False, and we want those that are not visited
                if self.adj_mat[node][neighbor] and not visited[neighbor]:
                    heapq.heappush(heap_q, (self.adj_mat[node][neighbor], neighbor, node))
        
        # once this is done, I have a list of the best tuples
        # I need the output in a mst adjacency matrix
        mst_adj_mat = np.zeros((num_nodes, num_nodes), dtype=float)

        # add the weights to the zero matrix, symmetric
        for weight, node1, node2 in best_tups:
            mst_adj_mat[node1][node2] = weight
            mst_adj_mat[node2][node1] = weight

        # assign it
        self.mst = mst_adj_mat
        self.tups = best_tups

    def return_weights(self):
        """
        If the construct_mst function has been run, you can call on this to produce the weights 
        for the mst
        """

        if self.tups is None:
            raise AttributeError("self.tups is not defined. Run the construct_mst function first")
        
        # sum of all the weights elements
        return np.sum([weights for weights, _, _ in self.tups])
        