import numpy as np


class ALU(object):

    def __init__(self):
        pass

    def __str__(self):
        s = ''
        return s

    def sigmoid(self, operand):
        return 1/(1 + np.exp(-operand))

    def gaussian(self, operand):
        pass

    def sqrt(self, operand):
        pass

    def sigmoid_symmetric(self, operand):
        pass

    def log(self, operand):
        pass

    def execute(self, operation, operand_1_val, operand_2_val):
        op = operation.op
        if op == '+':
            return operand_1_val + operand_2_val
        elif op == '-':
            return operand_1_val - operand_2_val
        elif op == '*':
            return operand_1_val * operand_2_val
        elif op == '/':
            return operand_1_val / operand_2_val
        elif op == '<':
            return operand_1_val < operand_2_val
        elif op == '<=':
            return operand_1_val <= operand_2_val
        elif op == '>':
            return operand_1_val > operand_2_val
        elif op == '>=':
            return operand_1_val >= operand_2_val
        elif op == '==':
            return operand_1_val == operand_2_val
        elif op == '!=':
            return operand_1_val != operand_2_val
        elif op == 'sigmoid':
            return self.sigmoid(operand_1_val)
        elif op == 'gaussian':
            return self.gaussian(operand_1_val)
        elif op == 'sqrt':
            return self.sqrt(operand_1_val)
        elif op == 'sigmoid_symmetric':
            return self.sigmoid_symmetric(operand_1_val)
        elif op == 'log':
            return self.log(operand_1_val)
        elif op == 'pass':
            return operand_1_val
        elif op == 'DONE':
            return 'DONE'
        else:
            raise Exception(f'ALU: Unsupported operation: {op}')


if __name__ == '__main__':
    from instruction import Operation

    op = Operation('+')

    alu = ALU()
    out_value = alu.execute(op, 2, 3)
    print(out_value)
