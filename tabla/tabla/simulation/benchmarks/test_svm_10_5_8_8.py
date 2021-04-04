from tablav2.simulation.simulator import Simulator
from tablav2.simulation.instruction import InstructionLoader


def test():
    benchmark_dir = '../../compilation_output/svm_wifi_10_5_8PU_8PE'
    config_file = '../configs/config_design_1.json'
    debug = False
    simulator = Simulator(benchmark_dir, config_file, debug)

    # simulator.only_debug_pu(0)

    # simulator.run_cycles(44)
    simulator.run()

    simulator.print_statistics()
    # simulator.write_statistics()


if __name__ == '__main__':
    test()
