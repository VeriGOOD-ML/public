from queue import Queue
import numpy as np

# This is a demonstration of Systolic Array matrix multiplication.
# We'll build the cell that makes up a systolic array processing element,
# then the array container that holds the array of cells.
# FIFOs will be implemented with Queue()

# First, we need to build the cell that will be tiled across the array
class SystolicArrayCell:
    def __init__(self):
        # For connection purposes, the cell needs to be able to find its
        # neighbors in the array. In real hardware, this would be done with
        # wiring and the X and Y position wouldn't matter
        self.pos_x = 0
        self.pos_y = 0

        # Each cell has the concept of a "partial sum" and an "activation".
        # These take one cycle to cross each cell (they would be delayed
        # with a register). To model this in python, we'll have a <field>
        # variable that represents the value driven by the neighboring cell,
        # and a <field>_out value representing the value driven by this cell.

        # partial sum: the running sum of the products, transmitted vertically
        self.partial_sum = 0
        self.partial_sum_out = 0
        # activation: the input activation value, transmitted horizontally
        self.activation = 0
        self.activation_out = 0

        # weight: The weight representing the second value to be multiplied
        self.weight = 0

        # Input fields, which will hold the connection to the cells or FIFOs
        # above and to the left of this cell
        self.input_activation = None
        self.input_partial_sum = None

    def set_psum(self, psum):
        self.partial_sum = psum
    # In the hardware implementation, we would use a control flow signal and
    # weight source via the partial sum lines (note that a weight is only half
    # the bits of that field, allowing control flow to be transmitted
    # alongside). For simplification here, we'll just say it's hacked in by
    # magic.
    def set_weight(self, weight):
        self.weight = weight

    # Connects this cell to its neighbors above and to the left
    def connect(self, pos_x, pos_y, array):
        self.pos_x = pos_x
        self.pos_y = pos_y

        # If we're at x position zero, then our left neighbor is a FIFO queue
        if self.pos_x is 0:
            self.input_activation = array.input[self.pos_y]
        # Otherwise, it's another cell
        else:
            self.input_activation = array.cells[self.pos_y][self.pos_x - 1]

        # If we're at y position zero, then our above neighbor is nothing
        if self.pos_y is 0:
            # All partial sums from here will just be 0
            self.input_partial_sum = None
        # Otherwise, our above neighbor is another cell
        else:
            self.input_partial_sum = array.cells[self.pos_y - 1][self.pos_x]

    # We'll model the transfer of signals through registers with a read() and a
    # compute() method.
    # read() represents the registers sampling data at the positive edge of the
    # clock
    def read(self):
        # Read the left neighbor
        # If this is a FIFO queue, take its value (or 0 if it's empty)
        if type(self.input_activation) is Queue:
            if self.input_activation.empty():
                self.activation = 0
            else:
                self.activation = self.input_activation.get()
        # If it is a cell, we read the value from activation_out
        else:
            self.activation = self.input_activation.activation_out

        # Read the above neighbor
        # If this is not connected, then the partial sum is always 0
        if self.input_partial_sum is None:
            self.partial_sum = 0
        # Otherwise, read the partial sum from the above cell
        else:
            self.partial_sum = self.input_partial_sum.partial_sum_out

    # compute() represents combinational logic that takes place between
    # positive edges of the clock (multiplication and addition)
    def compute(self):
        # First, the weight and activation in are multiplied
        product = self.weight * self.activation
        # Then that value is added to the partial sum from above and transmitted
        # downwards
        self.partial_sum_out = self.partial_sum + product
        # And the activation is transmitted to the right
        self.activation_out = self.activation

class Buffer(object):
    def __init__(self, buff_size, width):
        self._width = width
        self._buff_size = buff_size
        self._height = buff_size // width
        self._values = np.zeros((self.height, self.width))

    def load_buff(self, values):
        if len(values) == self.height:
            assert len(values[0]) == self.width
            self.values[:,:] = values
        else:
            assert len(values) == self.buff_size
            self.values[:,:] = np.asarray(values).reshape((self.height, self.width))

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def buff_size(self):
        return self._buff_size

    @property
    def values(self):
        return self._values

    def read(self, row_idx, num_rows):
        return self.values[row_idx: row_idx + num_rows, :]


# This represents our entire array: cells, source, and dest
class SystolicArray:
    # We'll take a parameter for the size of the square arrays to be multiplied
    def __init__(self, array_size, ibuf_size=None, wbuf_size=None, obuf_size=None):
        self._ibuf_size = ibuf_size or array_size * array_size * 4
        self._ibuf = Buffer(self.ibuf_size, array_size)

        self._wbuf_size = wbuf_size or array_size * array_size * 4
        self._wbuf = Buffer(self.wbuf_size, array_size)

        self._obuf_size = obuf_size or array_size * array_size * 4
        self._obuf = Buffer(self.obuf_size, array_size)

        self.array_size = array_size
        self.curr_cycle = 0

        # "cells" will hold the array of processing elements
        self.cells = []
        # This array is a square with dimensions "array_size"
        for _ in range(self.array_size):
            row = []
            for _ in range(self.array_size):
                cell = SystolicArrayCell()
                row.append(cell)
            self.cells.append(row)
        self.out_width =  3*self.array_size - 2
        self.offset = (self.out_width - self.array_size) // 2
        self.out_translations = {}
        for row in range(self.array_size):
            for col in range(self.array_size):
                self.out_translations[((self.offset + row + col), col)] = (row, col)
        # The source and dest will both be FIFO queues
        self.input = [Queue() for _ in range(self.array_size)]
        # self.output = [Queue() for _ in range(self.array_size)]
        self.output = np.zeros((self.array_size, self.array_size))

        # When all cells and source are created, then they can be connected
        # (again, this would be accomplished with wiring)
        for row_num, row in enumerate(self.cells):
            for col_num, cell in enumerate(row):
                cell.connect(col_num, row_num, self)

    # Accept a 2d array of weights, and "hack" them in. The hardware way to
    # fill weights is interesting but outside the scope of this demo.
    def fill_weights(self, weights):
        for row_num, row in enumerate(weights):
            for col_num, weight in enumerate(row):
                self.cells[row_num][col_num].set_weight(weight)

    # Accept a 2d array of activations.
    def fill_activations(self, activations):
        # For the systolic array to function properly, the activations must be
        # padded with a triangle of zeroes
        for row_num in range(self.array_size):
            for _ in range(row_num):
                self.input[row_num].put(0)

        # And the activations must be transposed before being added to the
        # input queue
        for row_num in range(self.array_size):
            col = [activations[x][row_num] for x in range(self.array_size)]
            for activation in col:
                self.input[row_num].put(activation)

    def fill_outputs(self, outputs):
        self.output = outputs

    def get_output_index(self, col, row):
        if (col, row) in self.out_translations.keys():
            return self.out_translations[(col, row)]
        else:
            return (-1,-1)
    # For this demo, all cells will read() the values of their neighbors first
    def read(self):
        for row in self.cells:
            for cell in row:
                cell.read()

    # And then after all cells have read(), they will compute() the next step
    def compute(self):
        for row in self.cells:
            for cell in row:
                cell.compute()

        # After each step of compute(), new dest will be but onto the output
        # queue

        for col_num in range(self.array_size):
            out_idx = self.get_output_index(self.curr_cycle, col_num)
            # print(self.curr_cycle, col_num)
            # print(out_idx)
            if out_idx[0] >= 0:
                self.output[out_idx[0], out_idx[1]] = self.cells[-1][col_num].partial_sum_out + self.output[out_idx[0], out_idx[1]]

    # Each cycle involves a read() and a compute()
    def cycle(self):
        # read() models register sampling on the positive edge of the clock
        self.read()
        # compute() models the combinational logic between clock edges
        self.compute()

    # run() will execute the array's computation, assuming it's been filled
    def run(self):
        # It takes 3n-2 cycles to compute the full matrix of results
        self.curr_cycle = 0
        for _ in range(3*self.array_size - 2):
            self.cycle()
            self.curr_cycle += 1

        return self.get_outputs()

    # The dest are also staggered and transposed, so we'll format them
    # before returning the results
    def get_outputs(self):
        return self.output

    @property
    def ibuf_size(self):
        return self._ibuf_size

    @property
    def wbuf_size(self):
        return self._wbuf_size

    @property
    def obuf_size(self):
        return self._obuf_size

    @property
    def ibuf(self):
        return self._ibuf

    @property
    def wbuf(self):
        return self._wbuf

    @property
    def obuf(self):
        return self._obuf

    def load_ibuf(self, values):
        self.ibuf.load_buff(values)

    def load_obuf(self, values):
        self.obuf.load_buff(values)

    def load_wbuf(self, values):
        self.wbuf.load_buff(values)

    def read_ibuf(self, start_index):
        read_values = self.ibuf.read(start_index, self.array_size)
        self.fill_activations(read_values)

    def read_wbuf(self, start_index):
        read_values = self.wbuf.read(start_index, self.array_size)
        self.fill_weights(read_values)

    def read_obuf(self, start_index):
        read_values = self.obuf.read(start_index, self.array_size)
        self.fill_outputs(read_values)

# # Here we'll use a small 3x3 test multiplication to see the systolic array
# # in action
# myArray = SystolicArray(3)
#
# activations = [
#     [1, 2, 3],
#     [4, 5, 6],
#     [7, 8, 9]
# ]
# myArray.fill_activations(activations)
#
# weights = [
#     [10, 20, 30],
#     [40, 50, 60],
#     [70, 80, 90]
# ]
# myArray.fill_weights(weights)
#
# res = myArray.run()
# assert (res == np.matmul(activations, weights)).all()
# print('Systolic array matches numpy matmul')
