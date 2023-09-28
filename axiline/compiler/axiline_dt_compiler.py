from pathlib import Path
from sklearn import tree
from axiline.compiler.decision_tree.DecisionTree import DecisionTree
from axiline.compiler.decision_tree.MappingNode import MappingNode
from axiline.compiler.decision_tree.BinaryTreeImpl import BinaryTreeImpl
from sklearn.datasets import load_digits
import argparse

def axiline_dt_compiler(decision_tree, num_unit, output_path, debug=False):
    if not isinstance(decision_tree,tree.DecisionTreeClassifier):
        exit(f"Error, decision_tree should be a sklearn tree.DecisionTree objective!")
    my_tree = DecisionTree(decision_tree).tree
    if debug:
        print(my_tree.properties)
    impl = BinaryTreeImpl(my_tree, num_unit, debug=debug)
    # cwd = Path(f"{__file__}").parent.parent
    output_path = f"{output_path}/dt_mapping.json"
    impl.to_json(output_path)


if __name__ == "__main__":
    iris = load_digits()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)
    num_unit = [10, 5, 3, 2, 2, 1, 1, 5]
    cwd = Path(f"{__file__}").parent.parent
    output_path = f"{cwd}/test/outputs/dt_mapping.json"

    axiline_dt_compiler(decision_tree=clf,
                        num_unit=num_unit,
                        output_path=output_path,
                        debug=True
     )
