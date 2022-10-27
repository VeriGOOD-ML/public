from binarytree import tree, bst, heap, Node
from axiline.compiler.decision_tree.DTNode import DTNode
from sklearn import tree as skl_tree
import numpy as np

class DecisionTree(Node):
    def __init__ (self, init_tree=None,debug=False):
        if isinstance(init_tree, skl_tree.DecisionTreeClassifier):
            self.from_sklearn_tree(init_tree,debug)

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, tree):
        if isinstance(tree, Node):
            self._tree = tree
        elif isinstance(tree, skl_tree.DecisionTreeClassifier):
            self.from_sklearn_tree(tree)


    def from_sklearn_tree (self, init_tree,debug=False):
        if not isinstance(init_tree, skl_tree.DecisionTreeClassifier):
            exit("not and sklearn tree object")
        n_nodes = init_tree.tree_.node_count
        children_left = init_tree.tree_.children_left
        children_right = init_tree.tree_.children_right
        feature = init_tree.tree_.feature
        threshold = init_tree.tree_.threshold

        node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
        is_leaves = np.zeros(shape=n_nodes, dtype=bool)
        root= DTNode(0)
        self.tree=root
        stack = [(0, 0, root)]  # start with the root node id (0) and its depth (0)
        while len(stack) > 0:
            # `pop` ensures each node is only visited once
            node_id, depth, c_node = stack.pop()

            # If the left and right child of a node is not the same we have a split
            # node
            is_split_node = children_left[node_id] != children_right[node_id]
            # If a split node, append left and right children and depth to `stack`
            # so we can loop through them
            c_node.feature = feature[node_id].item()
            c_node.threshold = threshold[node_id].item()
            if is_split_node:
                children_left_id=children_left[node_id].item()
                children_right_id = children_right[node_id].item()
                if children_left_id>0:
                    cl_node=DTNode(children_left_id)
                    c_node.left=cl_node
                if children_right_id>0:
                    cr_node=DTNode(children_right_id)
                    c_node.right = cr_node
                stack.append((children_left[node_id], depth + 1, cl_node))
                stack.append((children_right[node_id], depth + 1,cr_node ))
            else:
                is_leaves[node_id] = True

        if(debug):
            for i in range(n_nodes):
                if is_leaves[i]:
                    print(
                        "{space}node={node} is a leaf node.".format(
                            space=node_depth[i] * "\t", node=i
                        )
                    )
                else:
                    print(
                        "{space}node={node} is a split node: "
                        "go to node {left} if X[:, {feature}] <= {threshold} "
                        "else to node {right}.".format(
                            space=node_depth[i] * "\t",
                            node=i,
                            left=children_left[i],
                            feature=feature[i],
                            threshold=threshold[i],
                            right=children_right[i],
                        )
                    )

if __name__=="__main__":
    test()