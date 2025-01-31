import pytest
import numpy as np
from mst import Graph
from sklearn.metrics import pairwise_distances

# added an arguement cause I already calculated this
def check_mst(adj_mat: np.ndarray, 
              mst: np.ndarray, 
              expected_weight: int, 
              allowed_error: float = 0.0001, 
              tups = None):
    """
    
    Helper function to check the correctness of the adjacency matrix encoding an MST.
    Note that because the MST of a graph is not guaranteed to be unique, we cannot 
    simply check for equality against a known MST of a graph. 

    Arguments:
        adj_mat: adjacency matrix of full graph
        mst: adjacency matrix of proposed minimum spanning tree
        expected_weight: weight of the minimum spanning tree of the full graph
        allowed_error: allowed difference between proposed MST weight and `expected_weight`

    TODO: Add additional assertions to ensure the correctness of your MST implementation. For
    example, how many edges should a minimum spanning tree have? Are minimum spanning trees
    always connected? What else can you think of?

    """

    def approx_equal(a, b):
        return abs(a - b) < allowed_error

    total = 0
    for i in range(mst.shape[0]):
        for j in range(i+1):
            total += mst[i, j]
    assert approx_equal(total, expected_weight), 'Proposed MST has incorrect expected weight'

    edges = adj_mat.shape[0]
    # how many edges should a minimum spanning tree have?
    assert (edges - 1) == np.sum(mst > 0)/2, "The MST should have N-1 edges for N nodes"

    # Are minimum spanning trees always connected?
    if tups != None:
        # add in node 0 since it is nver included in the resultant node
        visited_nodes = set([0] + [nodes for _, _, nodes in tups])
        all_nodes = set(range(mst.shape[0]))
        assert visited_nodes == all_nodes, "Not all nodes were visited!"

    # are any nodes visited more than once?
    if tups != None:
        # add in node 0 since it is nver included in the resultant node
        visited_nodes = set([0] + [nodes for _, _, nodes in tups])
        visited_len = len(tups) + 1
        assert visited_len == len(visited_nodes), "Some nodes were visited more than once!"


def test_mst_small():
    """
    
    Unit test for the construction of a minimum spanning tree on a small graph.
    
    """
    file_path = './data/small.csv'
    g = Graph(file_path)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 8, tups=g.tups)


def test_mst_single_cell_data():
    """
    
    Unit test for the construction of a minimum spanning tree using single cell
    data, taken from the Slingshot R package.

    https://bioconductor.org/packages/release/bioc/html/slingshot.html

    """
    file_path = './data/slingshot_example.txt'
    coords = np.loadtxt(file_path) # load coordinates of single cells in low-dimensional subspace
    dist_mat = pairwise_distances(coords) # compute pairwise distances to form graph
    g = Graph(dist_mat)
    g.construct_mst()
    check_mst(g.adj_mat, g.mst, 57.263561605571695, tups=g.tups)


def test_mst_student():
    """
    
    TODO: Write at least one unit test for MST construction.
    
    """
    # check that it fails when it is not a NxN graph
    with pytest.raises(IndexError):
        file_path = './data/small_bad.csv'
        g = Graph(file_path)
        g.construct_mst()

    # check that it fails when the graph is not symmetric
    # written so it selects nodes based on the initial input
    # mst determined by hand (starting at 0) and compared to functional output
    true_mst = np.array([[0., 5., 0., 0.],
                         [5., 0., 1., 0.],
                         [0., 1., 0., 4.],
                         [0., 0., 4., 0.]])
    file_path = './data/small_nsym.csv'
    g = Graph(file_path)
    g.construct_mst()
    assert np.all(g.mst == true_mst), "mst differs from truth for non-symmetric matrix"

    # check what happens when you put in an empty, from edited initialization og Graph
    file_path = './data/small_empty.csv'
    with pytest.raises(ValueError):
        g = Graph(file_path)
