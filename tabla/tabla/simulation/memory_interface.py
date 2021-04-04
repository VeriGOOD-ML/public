import math
import os

from pathlib import Path

from .defaults import DEFAULT_AXI_DATA_WIDTH, DEFAULT_BURST_SIZE, DEFAULT_NUM_AXI


class MemoryInterface(object):

    def __init__(self, config, benchmark_dir):
        self.config = config

        self.benchmark_dir = benchmark_dir

        # Bitwidth of each data point (e.g. 16 bits)
        self.data_bitwidth = self.config['input_bitwidth']

        self.num_axi = DEFAULT_NUM_AXI

        # AXI width (e.g. 64 bits)
        self.axi_data_width = DEFAULT_AXI_DATA_WIDTH

        # Number of bursts in a single read by a single AXI from DDR (e.g. 16)
        self.burst_size_per_axi = DEFAULT_BURST_SIZE

        # Total number of bits read by a single AXI in a single burst
        self.burst_read_per_axi_bits = int(self.axi_data_width * self.burst_size_per_axi)

        # Number of data values read in a single axi burst read
        self.data_items_per_burst = int(self.burst_read_per_axi_bits / self.data_bitwidth)

        # Number of input data values
        self.num_input_data = self.get_num_input_data()

        # Number of weight values
        self.num_weight_data = self.get_num_weight_data()


        self.num_lanes_per_axi = int(self.axi_data_width / self.data_bitwidth)


    def __str__(self):
        s = '[Memory Interface Configuration]\n' + \
            f'AXI DATA WIDTH: {self.axi_data_width}\n' + \
            f'Burst size: {self.burst_size_per_axi}\n' + \
            f'Num AXI: {self.num_axi}\n' +\
            f'data items per burst: {self.data_items_per_burst}'
        return s


    def get_num_input_data(self):
        """
        Return the number of input data values for the given benchmark.
        """
        # Read input_data.txt file
        files = os.listdir(self.benchmark_dir)
        for file in files:
            if '_input_data.txt' in file:
                input_data_file = f'{Path(self.benchmark_dir)}/{file}'

                with open(input_data_file, 'r') as f:
                    input_data_lines = f.readlines()

                    return len(input_data_lines)


    def get_num_weight_data(self):
        """
        Return the number of weight values for the given benchmark.
        """
        # Read input_weights.txt file
        files = os.listdir(self.benchmark_dir)
        for file in files:
            if '_input_weights.txt' in file:
                input_weights_file = f'{Path(self.benchmark_dir)}/{file}'

                with open(input_weights_file, 'r') as f:
                    input_weights_lines = f.readlines()

                    return len(input_weights_lines)


    def get_offchip_access_stats(self):
        """
        Return a dictionary of accesses to AXI and memory lanes.
        """
        offchip_access_stats = {'read': 0,
                                'write': 0}

        data_items_per_burst_all_axi = self.data_items_per_burst

        num_bursts = math.ceil(self.num_input_data / data_items_per_burst_all_axi)
        num_bursts_weights = math.ceil(self.num_weight_data / data_items_per_burst_all_axi)


        # offchip_access_stats['read'] = num_bursts_weights + num_bursts

        offchip_access_stats['read'] = num_bursts_weights + num_bursts * 7703
        offchip_access_stats['write'] = num_bursts_weights
        return offchip_access_stats
