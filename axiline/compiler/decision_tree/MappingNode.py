import binarytree
import itertools
import copy

class MappingNode():
    id_iter = itertools.count()
    def __init__(self,depth=0):
        self.id = next(MappingNode.id_iter)
        self.successors=[]
        self.successor={}
        self.depth=depth

    def map(self,mapping_list,num_unit=0):
        if not isinstance(num_unit, int):
            exit("Error, numUnit should be a integer in mappingNode initiation!")
        if len(mapping_list) <= num_unit and num_unit>0:
            self.mapping = mapping_list
            self.num_unit = num_unit
        else:
            print(f"length of mapping list is {len(mapping_list)}; num of unit is {num_unit}")
            exit("Error mapping failed!")

    def compile(self,num_unit):
        queue = [self, 0]
        stage=0
        mapping={}
        if isinstance(num_unit,list):
            mapping_num_unit=num_unit[stage]
        else:
            mapping_num_unit=num_unit
        mapping_stage=self.mapping_compile(mapping_num_unit,[])
        while len(queue) > 1:
            node = queue.pop(0)
            if isinstance(num_unit, list):
                mapping_num_unit = num_unit[stage]
            else:
                mapping_num_unit = num_unit
            if node == 0:
                # store and init next stage
                mapping[stage]=mapping_stage
                stage+=1
                mapping_stage=self.mapping_compile(mapping_num_unit,[])
                queue.append(0)
            else:
                mapping_stage = self.mapping_compile(mapping_num_unit, node.mapping, mapping_stage)
                if len(node.successors):
                    queue.extend(node.successors)
        mapping[stage] = mapping_stage
        return mapping

    def print_mapping_summary(self,mapping):
        for key in mapping.keys():
            stage=mapping[key]
            unit=stage["num_unit"]
            print(f"Stage {key} has {unit} units, loading is")
            for mkey in stage.keys():
                if isinstance(mkey,int):
                    unit=stage[mkey]
                    print(f"{len(unit['nodes'])}")

    def __str__(self):
        queue=[self,0]
        s=''
        while len(queue)>1:
            node=queue.pop(0)
            if node==0:
                s+="------------------------\n"
                queue.append(0)
            else:
                s += f"{str(node.id)}:{node.mapping}\n"
                # print the address
                # if len(node.successor.keys()):
                #     for key in node.successor.keys():
                #         s += f" {key}-> {str(node.successor[key].id)}\n"
                if len(node.successors):
                    queue.extend(node.successors)
        return s

    def __getitem__(index: int):
        return successors[index]

    def max_depth(self):
        queue=[self,0]
        depth=1
        while len(queue)>1:
            node=queue.pop(0)
            if node==0:
                depth += 1
                queue.append(0)
            elif len(node.successors):
                queue.extend(node.successors)
        return depth

    def add_successor(self,successor, condition):
        if  not isinstance(successor,MappingNode):
            exit("Error, successor must be a mapping node!")

            # check if s contains only allowed characters
        allowed_s = "01x"
        if not all(ch in allowed_s for ch in condition):
            exit("Error, condition must be string with 0, 1, x !")
        else:
            self.successors.append(successor)
            self.successor[condition]=successor
            # print("successor added!")
            # print(f"successor list length {len(self.successors)}")

    def add_predecessor(self, predecessor):
        if not isinstance(predecessor,MappingNode):
            exit("Error, successor must be a mapping node!")
        self.predecessor=predecessor

    def mapping_compile(self, num_unit,  mapping_list, mapping_dict=None):
        init={
            "nodes": [],
            "features": [],
            "thresholds": [],
            "memory_depth": 0
        }
        if not mapping_dict:
            mapping_dict = {}
            mapping_dict["num_unit"] = 0
            for i in range(len(mapping_list)):
                mapping_dict[i]=copy.deepcopy(init)
        if mapping_dict["num_unit"]<len(mapping_list):
            mapping_dict["num_unit"]=len(mapping_list)
        for i, node in enumerate(mapping_list):
            if i not in mapping_dict.keys():
                mapping_dict[i]=copy.deepcopy(init)
            mapping_dict[i]['nodes'].append(node.value)
            mapping_dict[i]['features'].append(node.feature)
            mapping_dict[i]['thresholds'].append(node.threshold)
            mapping_dict[i]['memory_depth']+=1
        return mapping_dict

def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError(f"index {index} outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]