import numpy as np

from _collections import deque

from .defaults import DEFAULT_NAMESPACE_BUFFER_SIZE, DEFAULT_INPUT_BITWIDTH
from .common import bitwidth_to_nptype
from . import Data


"""
Buffer represents PE buffers, (so-called Namespaces)
"""
class Buffer(object):
    # Initial data value for each item in this buffer
    INIT_VALUE = 0

    def __init__(self, name, size=DEFAULT_NAMESPACE_BUFFER_SIZE, bitwidth=DEFAULT_INPUT_BITWIDTH, is_loaded=False, pe_id=None):
        # Buffer name (e.g. NW, NI...)
        self.name = name

        # Number of data items in this buffer
        self.size = size

        # Bitwidth of each data item
        self.bitwidth = bitwidth

        # Items in this buffer
        if is_loaded:
            self.entries_ = [Data(value=0, abs_pe_id=pe_id, rel_pe_id=pe_id) for _ in range(self.size)]
        else:
            self.entries_ = [None] * size
        self.items = np.full(self.size, 0, dtype=bitwidth_to_nptype[self.bitwidth])

        # For each data item, set a flag if new data has been written to the index of the item
        # TODO Not a good style
        # TODO Change True to False

        if type(self.size) == int:
            self.data_written_flag = [is_loaded] * self.size
        else:
            self.data_written_flag = [is_loaded] * self.size[0]

        # For push and pop
        # self.head_index = 0
        # self.tail_index = 0

        # self.new_data_written_count = 0

    def __str__(self):
        s = f"{self.name}, size: {self.size}, bitwidth: {self.bitwidth}"
        return s

    def read(self, index):
        """
        Read data value from given index.
        TODO: Add array index out of bound checks
        """
        if self.data_written_flag[index] == False:
            return None
        return self.entries_[index].value
        # return self.items[index].value

    def write(self, index, value, abs_pe_id, rel_pe_id):
        """
        Write data value to the given index.
        TODO: Add array index out of bound checks
        """
        self.entries_[index] = Data(value=value, abs_pe_id=abs_pe_id, rel_pe_id=rel_pe_id)
        self.items[index] = value
        self.data_written_flag[index] = True

    def peek(self, index):
        """
        Return True if data is available at the given index.
        """
        return self.data_written_flag[index]


    # TODO push(), pop() and self.new_data_written_count should belong to a
    # different class. Make a new class called FIFO for these.

    # def push(self, value):
    #     """
    #     Only to be used by PENB Read Buffer (FIFO).
    #     """

    #     if self.tail_index < self.size:
    #         self.write(self.tail_index, value)
    #         self.new_data_written_count += 1
    #         self.tail_index += 1

    # def pop(self):
    #     """
    #     Pop the first element from this buffer. Only to be used by PENB Read Buffer (FIFO).
    #     """


    #     data = self.read(self.head_index)
    #     self.new_data_written_count -= 1
    #     self.head_index += 1
    #     return data

    # # TODO refactor this with push() and pop() to FIFO class
    # def peek_fifo(self):
    #     if self.new_data_written_count > 0:
    #         return True
    #     else:
    #         return False


class FIFO(object):

    def __init__(self, name, size=DEFAULT_NAMESPACE_BUFFER_SIZE, bitwidth=DEFAULT_INPUT_BITWIDTH):
        # Buffer name (e.g. NW, NI...)
        self.name = name

        # Number of data items in this buffer
        self.size = size

        # Bitwidth of each data item
        self.bitwidth = bitwidth

        # Items in this buffer
        self.items = np.full(self.size, 0, dtype=bitwidth_to_nptype[self.bitwidth])

        # self.entries = deque(self.items)
        self.entries = deque()


    def push(self, value, abs_pe_id, rel_pe_id):
        data = Data(value=value, abs_pe_id=abs_pe_id, rel_pe_id=rel_pe_id)
        self.entries.append(data)
        return True

    def pop(self):

        return self.entries.popleft()

    def peek_fifo(self):
        if len(self.entries) > 0:
            return True
        else:
            return False


"""
Special type of buffer for PEGB Write Buffer. This needs to store both data value
and destination PE ID (relative).
"""
class PEGBWriteBuffer(FIFO):

    def __init__(self, name, size=DEFAULT_NAMESPACE_BUFFER_SIZE, bitwidth=DEFAULT_INPUT_BITWIDTH):
        # Create a two-dimensional np array, where the second dimension stores destination PE ID
        shape = [size, 2]
        super(PEGBWriteBuffer, self).__init__(name, shape, bitwidth)

    # def push(self, value):
    #     """
    #     Push value to tail of this buffer.
    #     """
    #     # self.size is a shape (num of entries, 2)
    #     if self.tail_index < self.size[0]:
    #         self.write(self.tail_index, value)
    #         self.new_data_written_count += 1
    #         self.tail_index += 1

    #         # TODO only for debug purposes - remove return statements after done
    #         return True
    #     return False

    # def push(self, value, abs_pe_id, rel_pe_id):
    #     if len(self.entries) > 0:
    #
    #         self.entries.append(Data(value=value, rel_pe_id=rel_pe_id, abs_pe_id=abs_pe_id))
    #         return True
    #     return False


"""
Special type of buffer for PUGB Write Buffer. This needs to store both data value
and destination PU ID.
"""
class PUGBWriteBuffer(FIFO):

    def __init__(self, name, size=DEFAULT_NAMESPACE_BUFFER_SIZE, bitwidth=DEFAULT_INPUT_BITWIDTH):
        shape = [size, 2]
        super(PUGBWriteBuffer, self).__init__(name, shape, bitwidth)

    # def push(self, value):
    #     """
    #     Push value to tail of this buffer.
    #     """
    #     # self.size is a shape (num of entries, 2)
    #     if self.tail_index < self.size[0]:
    #         self.write(self.tail_index, value)
    #         self.new_data_written_count += 1
    #         self.tail_index += 1

    #         # TODO only for debug purposes - remove return statements after done
    #         return True
    #     return False

    # def push(self, value, abs_pe_id, rel_pe_id):
    #     if len(self.entries) > 0:
    #         self.entries.append(Data(value=value, rel_pe_id=rel_pe_id, abs_pe_id=abs_pe_id))
    #         return True
    #     return False


if __name__ == '__main__':
    nw = Buffer('NW', bitwidth=8)
    nw.write(0, 2)
    nw.write(1, 3)
    nw.write(2, 7)
    print(nw)

    data = nw.read(1)
    print(data)
    print(type(data))

    # ff = FlipFlop('inst_flip_flop')
    # print(ff)
    # ff.write(5)
    # data = ff.read()
    # print(data)
