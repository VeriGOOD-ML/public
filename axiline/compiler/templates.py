

class Templates:
    def __init__(self,init=0):
        self.operations = []
        self.names = []
        self.high_level = []
        self.low_level = []
        self.data = {}
        if isinstance(init, dict):
            self.load(init)
        elif isinstance(init, list):
            for element in init:
                if isinstance(element, dict):
                    self.load(element)



    def load(self,template):
        if "name" in template.keys() and 'operation' in template.keys():
            self.names.append(template['name'])
            self.operations.append(template['operation'])

            if template['level']==2:
                # hign level pattern include more than one high-level DFG nodes
                end=template['operation'][-1]
                nodes=template['operation']
                name=template['name']
                pattern=Pattern(end,nodes,name)
                self.high_level.append(pattern)
            # elif template['level']==1:
            #     # middle level pattern include one high-level DFG node
            #     end = template['operation'][-1]
            #     nodes = template['operation']
            #     name = template['name']
            #     pattern = Pattern(end, nodes, name)
            #     self.middle_level.append(pattern)
            elif template['level']<2:
                # low level pattern include one basic operation
                end = template['operation'][-1]
                nodes = template['operation']
                name = template['name']
                pattern = Pattern(end, nodes, name)
                self.low_level.append(pattern)
            self.data[template['name']]={
                'name':template['name'],
                'operation':template['operation'],
                'level':template['level']
            }
            if 'input' in template.keys():
                self.data[template['name']]['input']=template['input']


class Pattern():
    def __init__(self, end='', nodes=[], name=''):
        self.end=end
        self.nodes=nodes
        self.name=name
        self.is_high=len(nodes)>1

    @property
    def end(self):
        #print("Getting value...")
        return self._end

    @property
    def nodes(self):
        # print("Getting value...")
        return self._nodes

    @property
    def name(self):
        # print("Getting value...")
        return self._name

    @end.setter
    def end(self, value):
        if isinstance(value, str):
            self._end = value

    @nodes.setter
    def nodes(self, value):
        if isinstance(value, list):
            self._nodes = value
            self.is_high = len(value) > 1

    @name.setter
    def name(self, value):
        if isinstance(value, str):
            self._name = value