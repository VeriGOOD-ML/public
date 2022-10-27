import binarytree
import itertools
import json
from axiline.compiler.decision_tree.MappingNode import MappingNode,replacer
import json

class BinaryTreeImpl:
    def __init__(self,tree=0, num_unit=0,debug=False):
        self.tree=tree
        self.num_unit=num_unit
        self.mapping_data={}
        if isinstance(tree,binarytree.Node):
            self.mapping(self.tree,self.num_unit,debug)

    def mapping(self, tree, p, debug=False):
        if not isinstance(tree, binarytree.Node):
            exit("Error. Tree should be a binarytree objective")
        if not (isinstance(p, int) or isinstance(p, list)):
            exit("Error. P should be a integer")

        #map = [[] for i in range(p)]
        map = {}
        impl_graph=MappingNode(0)
        branches = [[tree,impl_graph]]
        addrs=[]
        # while len(branches):
        while len(branches):
            addr = {}
            branch,m_node=branches.pop(0)
            node_list = list(branch)
            if isinstance(p, int):
                num_unit=p
            elif isinstance(p, list):
                if len(p)<=m_node.depth:
                    exit("num_unit defination smaller than the number of pipeline stages!")
                num_unit=p[m_node.depth]
            if num_unit < len(node_list):
                current = node_list[0:num_unit]
            else:
                current = node_list
            m_node.map(current, num_unit)
            for i, node in enumerate(current):
                if i in map.keys():
                    map[i].append(node)
                else:
                    map[i]=[node]
            addr = self.leaf_search(branch, current, addr, debug)
            successor_depth=m_node.depth+1
            for i, key in enumerate(addr.keys()):
                m_node.add_successor(MappingNode(successor_depth),key)
                branches.append([addr[key],m_node.successors[i]])
            addrs.append(addr)

        self.addrs=addrs
        self.impl_graph=impl_graph
        self.mapping_data=impl_graph.compile(p)

        if debug:
            print(f"Final Mapping Data:")
            print(addrs)
            print(f"Final Mapping:")
            print(impl_graph)
            print(f"max depth of the impl graph is {impl_graph.max_depth()}")
            print(self.mapping_data)
            impl_graph.print_mapping_summary(self.mapping_data)

    def to_json(self,dir=None):
        with open(dir, "w") as outfile:
            json.dump(self.mapping_data, outfile,indent = 4)

    def dump(self,dir):
        if dir.endswith('.json'):
            with open(f"{dir}", 'w') as fp:
                json.dump(self.addrs, fp)
        else:
            with open(f"{dir}/mapping.json", 'w') as fp:
                json.dump(self.addrs, fp)


    def leaf_search(self, branch: binarytree.Node, node_list: list, addr: dict, debug) -> dict:
        length = len(node_list)
        template = "x" * length
        for node in node_list:
            if node.max_leaf_depth>0:
                if node.left not in node_list and node.right not in node_list:
                    index =replacer(template, '*',node_list.index(node))
                    addr = self.backward_trace(node, branch,node_list, addr, index,debug)
                elif node.left not in node_list:
                    index =replacer(template, '#',node_list.index(node))
                    addr = self.backward_trace(node, branch,node_list, addr, index,debug)
                elif node.right not in node_list:
                    index = replacer(template, '%', node_list.index(node))
                    addr = self.backward_trace(node, branch, node_list, addr, index,debug)
        return addr

    def backward_trace(self, node:binarytree.Node, branch:binarytree.Node, node_list: list, addr:dict, idx: str,debug)->dict:
        # if debug:
        #     print(f"current node \n {node}")
        #     print(f"current branch \n {branch}")
        #     print(f"current node list\n {node_list}")
        #     print(f"current index\n {idx}")

        if node==branch:
            # print(binarytree.get_index(branch,node))
            if '*' in idx:
                idxl=idx.replace('*', '0')
                idxr=idx.replace('*', '1')
                #print(idx.index('*'))
                addr[idxl] =node_list[idx.index('*')].left
                addr[idxr] = node_list[idx.index('*')].right
            elif '#' in idx :
                idxl=idx.replace('#', '0')
                addr[idxl] = node_list[idx.index('#')].left
            elif '%' in idx :
                idxr=idx.replace('%', '1')
                addr[idxr] = node_list[idx.index('%')].right
            return addr
        else:
            parent=binarytree.get_parent(branch,node)
            if parent in node_list:
                if node==parent.left:
                    idx= replacer(idx,'0', node_list.index(parent))
                else:
                    idx = replacer(idx, '1', node_list.index(parent))
                addr=self.backward_trace(parent, branch, node_list,addr, idx, debug)
            return addr
