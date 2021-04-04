import copy

class Node(object):
    """
    Base class for node
    """

    counter = 0

    def __init__(self, index=None):

        # index is a unique value for all the nodes created
        if index:
            self.index = index
            Node.counter = index + 1
        else:
            Node.counter += 1
            self.index = Node.counter
        
        # connectivity
        self._succs = {}
        self._preds = {}
        
        # attributes
        self._attrs = {}
        self._attrs['in_degree'] = 0
        self._attrs['out_degree'] = 0
    
        # graph
        self._graph = None

    def __str__(self):
        return f'node {self.index}: \
                 preds={self.get_preds_indices()} ({self._attrs["in_degree"]}), \
                 succs={self.get_succs_indices()} ({self._attrs["out_degree"]})'

    def delete(self):
        
        # remove all connections
        for succ in self.get_succs():
            self.remove_succ(succ)
        for pred in self.get_preds():
            self.remove_pred(pred)

        # remove from graph
        if self._graph:
            self._graph.remove_node(self)
        
        # delete self
        del self

    def get_all_attributes(self):
        att_list = []
        for k, v in self._attrs.items():
            att_list.append({"key": k, "value": v})
        return att_list

    def dissolve(self):
        # make connections between preds and succs
        for pred in self.get_preds():
            for succ in self.get_succs():
                pred.add_succ(succ)

        # delete self
        self.delete()

    # connectivity
    def add_succ(self, dst_node):
        self._add_succ(dst_node)
        dst_node._add_pred(self)
        assert self.get_attr('out_degree') == len(self.get_succs()), 'out_degree and length of succs should be same'

    def add_pred(self, src_node):
        self._add_pred(src_node)
        src_node._add_succ(self)
        assert self.get_attr('in_degree') == len(self.get_preds()), 'in_degree and length of preds should be same'
    
    def add_succ_by_index(self, dst_node_index:int):
        assert self._graph != None, 'add_succ_by_index function cannot be used without setting graph'
        dst_node = self._graph.get_node_by_index(dst_node_index)
        self.add_succ(dst_node)
        assert self.get_attr('out_degree') == len(self.get_succs()), 'out_degree and length of succs should be same'

    def add_pred_by_index(self, src_node_index:int):
        assert self._graph != None, 'add_pred_by_index function cannot be used without setting graph'
        src_node = self._graph.get_node_by_index(src_node_index)
        self.add_pred(src_node)
        assert self.get_attr('in_degree') == len(self.get_preds()), 'in_degree and length of preds should be same'

    def _add_succ(self, dst_node):
        self._succs[dst_node.index] = dst_node
        self.increment_out_degree()

    def _add_pred(self, src_node):
        self._preds[src_node.index] = src_node
        self.increment_in_degree()

    def remove_succ(self, dst_node):
        self._remove_succ(dst_node)
        dst_node._remove_pred(self)
        assert self.get_attr('out_degree') == len(self.get_succs()), 'out_degree and length of succs should be same'

    def remove_pred(self, src_node):
        self._remove_pred(src_node)
        src_node._remove_succ(self)
        assert self.get_attr('in_degree') == len(self.get_preds()), 'in_degree and length of preds should be same'

    def remove_succ_by_index(self, dst_node_index:int):
        assert self._graph != None, 'remove_succ_by_index function cannot be used without setting graph'
        assert dst_node_index in self._succs, 'dst_node_index is not in succs'
        dst_node = self._graph.get_node_by_index(dst_node_index)
        self.remove_succ(dst_node)
        assert self.get_attr('out_degree') == len(self.get_succs()), 'out_degree and length of succs should be same'

    def remove_pred_by_index(self, src_node_index:int):
        assert self._graph != None, 'remove_pred_by_index function cannot be used without setting graph'
        assert src_node_index in self._preds, 'src_node_index is not in preds'
        src_node = self._graph.get_node_by_index(src_node_index)
        self.remove_pred(src_node)
        assert self.get_attr('in_degree') == len(self.get_preds()), 'in_degree and length of preds should be same'
    
    def _remove_succ(self, dst_node):
        assert dst_node.index in self._succs, 'dst_node is not in succs'
        del self._succs[dst_node.index]
        self.decrement_out_degree()

    def _remove_pred(self, src_node):
        assert src_node.index in self._preds, 'src_node is not in preds'
        del self._preds[src_node.index]
        self.decrement_in_degree()


    def increment_in_degree(self):
        self._attrs['in_degree'] += 1
        assert self._attrs['in_degree'] >= 0, 'in_degree needs to be positive'
       
    def decrement_in_degree(self):
        self._attrs['in_degree'] -= 1
        assert self._attrs['in_degree'] >= 0, 'in_degree needs to be positive'
 
    def increment_out_degree(self):
        self._attrs['out_degree'] += 1
        assert self._attrs['out_degree'] >= 0, 'out_degree needs to be positive'
 
    def decrement_out_degree(self):
        self._attrs['out_degree'] -= 1
        assert self._attrs['out_degree'] >= 0, 'out_degree needs to be positive'
 

    # NOTE Python 2.x calling keys makes a copy of the key that you can iterate over while modifying the dict.
    # however, this doesn't work in Python 3.x because keys returns an iterator instead of a list.
    # REF: https://stackoverflow.com/questions/11941817/how-to-avoid-runtimeerror-dictionary-changed-size-during-iteration-error
    def get_succs(self):
        return list(self._succs.values())

    def get_preds(self):
        return list(self._preds.values())

    def get_succs_indices(self):
        return list(self._succs.keys())

    def get_preds_indices(self):
        return list(self._preds.keys())

    def get_succ_by_index(self, node_index:int):
        return self._succs[node_index]

    def get_pred_by_index(self, node_index:int):
        return self._preds[node_index]
    
    def is_attr_key(self, attr_key:str):
        return attr_key in self._attrs

    # attributes
    def set_attr(self, attr_key:str, attr_value):
        self._attrs[attr_key] = attr_value

    def get_attr(self, attr_key:str):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        return self._attrs[attr_key]

    def append_attr(self, attr_key:str, attr_value):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        assert type(self._attrs[attr_key]) == list, 'attr_key is not list'
        if type(attr_value) == list:
            self._attrs[attr_key].extend(attr_value)
        else:
            self._attrs[attr_key].append(attr_value)

    def duplicate_attr_by_key(self, old_attr_key:str, new_attr_key:str):
        assert old_attr_key in self._attrs, 'old_attr_key is not in attrs'
        
        # this ought to be deep copy
        self._attrs[new_attr_key] = copy.deepcopy(self._attrs[old_attr_key])        

    def clear_attrs(self):
        self._attrs = {}

    def clear_attr_by_key(self, attr_key:str):
        assert attr_key in self._attrs, 'attr_key is not in attrs'
        del self._attrs[attr_key]


    def set_graph(self, graph):
        self._graph = graph

    def clear_graph(self):
        self._graph = None

    
    def get_all_succs(self, leaf_condition=lambda x: False, filter_condition=lambda x: True, debug=False):

        # create output set
        out = []

        # create stack
        # NOTE stack means this is DFS
        stack = []
        curr_path = [self]
        stack.append(curr_path)

        # while stack is not empty
        while stack:
            if debug:
                print([node.index for node in curr_path])
            
            # if inside stack
            curr_path = stack.pop()
            
            # if all nodes have been visited, add to outs
            if curr_path[-1] != self and leaf_condition(curr_path[-1]):
                filtered_nodes = [node for node in curr_path if filter_condition(node)]
                out.extend(filtered_nodes)
                continue
            
            # append next_path with succ to stack
            for succ in curr_path[-1].get_succs():
                next_path = copy.copy(curr_path + [succ]) # shallow copy
                stack.append(next_path)
        
        # NOTE shallow copy
        return list(dict.fromkeys(out))
        
    def get_all_preds(self, leaf_condition=lambda x: False, filter_condition=lambda x: True, debug=False):

        # create output set
        out = []

        # create stack
        # NOTE stack means this is DFS
        stack = []
        curr_path = [self]
        stack.append(curr_path)

        # while stack is not empty
        while stack:
            if debug:
                print([node.index for node in curr_path])
            
            # if inside stack
            curr_path = stack.pop()
            
            # if all nodes have been visited, add to outs
            if curr_path[-1] != self and leaf_condition(curr_path[-1]):
                filtered_nodes = [node for node in curr_path if filter_condition(node)]
                out.extend(filtered_nodes)
                continue
            
            # append next_path with pred to stack
            for pred in curr_path[-1].get_preds():
                next_path = copy.copy(curr_path + [pred]) # shallow copy
                stack.append(next_path)
        
        # NOTE shallow copy
        return list(dict.fromkeys(out))
        
