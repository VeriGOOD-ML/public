from queue import Queue
import numpy as np
from pprint import pprint

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
        self.index_translations = {}
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


# This represents our entire array: cells, source, and dest
class SystolicArray:
    # We'll take a parameter for the size of the square arrays to be multiplied
    def __init__(self, array_size):
        self.array_size = array_size
        self.ccycle = 0
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
        pprint(self.out_translations)

        # The source and dest will both be FIFO queues
        self.input = [Queue() for _ in range(self.array_size)]
        self.output = [Queue() for _ in range(self.array_size)]
        self.temp_out = np.zeros((self.array_size, self.array_size))
        self.tout = []

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
        self.tout.append([])
        for col_num in range(self.array_size):
            psum = self.cells[-1][col_num].partial_sum_out
            self.output[col_num].put(psum)
            self.tout[-1].append(psum)

            cmp_row = self.get_output_row(self.ccycle, col_num)
            if cmp_row[0] >= 0:
                print(f"Column number:  {col_num}\tRow num: {self.ccycle}\n"
                      f"Correct column: {cmp_row[0]}\tCorrect row: {cmp_row[1]}\n"
                      f"Value: {self.cells[-1][col_num].partial_sum_out}")
                print("\n")
        print(f"Current cycle: {self.ccycle}\n Output:\n {np.asarray(self.tout).T}")

    def get_output_row(self, col, row):

        if (col, row) in self.out_translations.keys():
            return self.out_translations[(col, row)]
        else:
            return (-1,-1)



    # Each cycle involves a read() and a compute()
    def cycle(self):
        # read() models register sampling on the positive edge of the clock
        self.read()
        # compute() models the combinational logic between clock edges
        self.compute()
    # run() will execute the array's computation, assuming it's been filled
    def run(self):
        # It takes 3n-2 cycles to compute the full matrix of results
        self.ccycle = 0
        for _ in range(3*self.array_size - 2):
            self.cycle()
            self.ccycle += 1


        return self.get_outputs()

    # The dest are also staggered and transposed, so we'll format them
    # before returning the results
    def get_outputs(self):
        ret = []

        # Remove the staggering by throwing away the appropriate number of 0's
        for col_num in range(self.array_size):
            for _ in range(col_num + self.array_size - 1):
                self.output[col_num].get()

        # And transpose the results
        for row_num in range(self.array_size):
            row = []
            for output_col in self.output:
                row.append(output_col.get())
            ret.append(row)

        return ret

# Here we'll use a small 3x3 test multiplication to see the systolic array
# in action
myArray = SystolicArray(6)

activations = [
    [1, 2, 3, 7, 8, 9],
    [4, 5, 6, 7, 8, 9],
    [7, 8, 9, 7, 8, 9],
    [1, 2, 3, 7, 8, 9],
    [4, 5, 6, 7, 8, 9],
    [7, 8, 9, 7, 8, 9],
]
myArray.fill_activations(activations)

weights = [
    [10, 20, 30, 70, 80, 90],
    [40, 50, 60, 70, 80, 90],
    [70, 80, 90, 70, 80, 90],
    [10, 20, 30, 70, 80, 90],
    [40, 50, 60, 70, 80, 90],
    [70, 80, 90, 70, 80, 90]
]
myArray.fill_weights(weights)

res = myArray.run()
print(f"Final output:\n{np.asarray(res)}")
assert (res == np.matmul(activations, weights)).all()
print('Systolic array matches numpy matmul')
