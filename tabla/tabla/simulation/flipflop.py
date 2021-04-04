from .defaults import DEFAULT_FLIP_FLOP_SIZE
from . import Data
from typing import List
from collections import namedtuple


class FlipFlop(object):

    def __init__(self, name, size=DEFAULT_FLIP_FLOP_SIZE):
        self.name = name
        self.written_data = False
        self.size = size
        self.data: List[Data] = []

        # self.data = [None]

    def __str__(self):
        s = f'{self.name}, size: {self.size}'
        return s

    def read(self, abs_pe_id):
        """
        BIG ASSUMPTION: data is set to None once it has been read. That way,
        we avoid keep reading the same data in the pipeline.
        """
        data = None
        for d in self.data:
            if d.abs_pe_id == abs_pe_id:
                data = d
                break
        if data is not None:
            self.data.remove(data)
            return data.value
        else:
        # data = self.data[0]
        # self.data[0] = None
            return data

    def write(self, value, abs_pe_id, rel_pe_id):
        # self.data[0] = value
        self.data.append(Data(value=value, abs_pe_id=abs_pe_id, rel_pe_id=rel_pe_id))

    def peek(self, abs_pe_id):
        """
        Return True if data is available
        """

        for d in self.data:
            if d.abs_pe_id == abs_pe_id:
                return True
        return False
        # if self.data[0] != None:
        #     return True
        # else:
        #     return False


if __name__ == '__main__':
    ff = FlipFlop('inst_flip_flop')
    print(ff)
    ff.write(5)
    data = ff.read()
    print(data)

    print(ff.data)
