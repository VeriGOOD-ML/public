import sys
sys.path.append('..')

from codelets.graph.node import Node
from codelets.graph.graph import Graph

if __name__ == '__main__':
    print('[TEST] test code for graph')

    # test 1
    graph1 = Graph()
    node1 = Node()
    node2 = Node()
    node3 = Node()

    print('       add edges 1-->2')
    print('       now 1-->2')
    graph1._add_node(node1)
    graph1._add_node(node2)
    graph1._add_node(node3)
    graph1._add_edge(node1, node2)

    print(node1)
    print(node2)

    print('       remove edge')
    print('       now 1   2')
    node1.remove_succ(node2)

    print(node1)
    print(node2)
    
    print('       add edges 1-->2-->3')
    print('       now 1-->2-->3')
    graph1._add_edge(node1, node2)
    graph1._add_edge(node2, node3)
    
    print(node1)
    print(node2)
    print(node3)

    print('       delete node1')
    print('       now 2-->3')
    node1.delete()

    print(node2)
    print(node3)
    
    print('       add edges 4-->2')
    print('       now 4-->2-->3')
    node4 = Node()
    graph1._add_node(node4)
    graph1._add_edge(node4, node2)
    
    # visualize
    graph1.visualize()
    
    print(node2)
    print(node3)
    print(node4)

    print('       delete node4')
    print('       now 2-->3')
    graph1.delete_node(node4)

    print(node2)
    print(node3)

    print('       add edges 3-->2')
    print('       now 2-->3-->2')
    graph1._add_edge(node3, node2)
    
    print(node2)
    print(node3)

    print('       delete node3')
    print('       now 2')
    node3.delete()

    print(node2)
    graph1.visualize("test1")

