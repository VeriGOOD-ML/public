from sklearn.datasets import load_boston, load_digits
from sklearn.datasets import load_iris
from pathlib import Path
from sklearn import tree
from axiline.compiler.decision_tree.DecisionTree import DecisionTree
from axiline.compiler.decision_tree.MappingNode import MappingNode
from axiline.compiler.decision_tree.BinaryTreeImpl import BinaryTreeImpl
from axiline.compiler.axiline_dt_compiler import axiline_dt_compiler
import graphviz

def test_dt():
    iris = load_digits()
    X, y = iris.data, iris.target
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X, y)

    my_tree=DecisionTree(clf).tree
    print(my_tree.properties)
    #print(my_tree)
    num_unit=[10,5,3,2,2,1,1,5]

    impl=BinaryTreeImpl(my_tree,num_unit,debug=True)
    cwd = Path(f"{__file__}").parent.parent
    output_path = f"{cwd}/test/outputs/dt_mapping.json"
    impl.to_json(output_path)
    print(tree.plot_tree(clf))

    # dot_data = tree.export_graphviz(clf, out_file=None)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")

    # dot_data = tree.export_graphviz(clf, out_file=None,
    #                       feature_names=iris.feature_names,
    #                       class_names=iris.target_names,
    #                       filled=True, rounded=True,
    #                     special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("iris")
