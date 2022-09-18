from codelets.adl.operation import Operation, Operand
from codelets.codelet_impl.codelet import Codelet


def collect_operation_dependencies(cdlt: Codelet, operation: Operation):
    all_dependencies = []
    for d in operation.dependencies:
        d_op = cdlt.op_map[d]
        all_dependencies += collect_operation_dependencies(cdlt, d_op)
    return all_dependencies + operation.dependencies.copy()

def collect_operand_dependencies(operand: Operand, cdlt: Codelet):
    operand_deps = []
    for d in operand.dependencies:
        d_op = cdlt.op_map[d]
        operand_deps += collect_operation_dependencies(cdlt, d_op)

    return list(set(operand_deps + operand.dependencies.copy()))
