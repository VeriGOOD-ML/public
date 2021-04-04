from .defaults import DEFAULT_BUS_BITWIDTH
from .common import bitwidth_to_nptype
from . import Data
from pytools import memoize_method


class Bus(object):

    def __init__(self, bitwidth=DEFAULT_BUS_BITWIDTH):
        self.bitwidth = bitwidth

        # Bus can only have a single value at a time.
        # bitwidth_to_nptype[self.bitwidth] returns numpy data types, e.g. np.short,
        # and the parantheses instantiates a data value of that type, with default value 0.
        # np.short() produces a value of type np.short.
        self.data = bitwidth_to_nptype[self.bitwidth]()


class PENB(Bus):
    """
    PE Neighbor Bus. Connects PE N to PE N + 1.
    """

    def __init__(self, source_pe, dest_pe, bitwidth=DEFAULT_BUS_BITWIDTH, debug=False):
        super(PENB, self).__init__(bitwidth)
        self.source_pe = source_pe
        self.dest_pe = dest_pe

        # Flag to be set if new data has been written to this bus. Set to False
        # once data is read (because the data is now in PENB Read Buffer).
        # Note: Since buses only have one value at a time, once data is read from
        # it, it won't be read again.
        self.new_data_written = False

        self.debug = debug

    def __str__(self):
        s = f'[PENB] PE {self.source_pe.relative_id} -> PE {self.dest_pe.relative_id}, ' + \
            f'data: {self.data}, ' + \
            f'bitwidth {self.bitwidth}'
        return s

    def read(self):
        self.new_data_written = False
        return self.data

    def write(self, data):
        """
        Write data value to this bus. PE N (source PE) sets the value of this PE Neighbor Bus.
        """
        assert isinstance(data, Data)
        self.data = data
        self.new_data_written = True

    def write_to_pe_read_buffer(self):
        """
        Write bus data to destination PE Read Buffer.
        """
        if self.new_data_written:
            read_buffer = self.dest_pe.penb_read_buffer
            read_buffer.push(self.data, self.dest_pe.absolute_id, self.dest_pe.relative_id)
            self.new_data_written = False
            if self.debug == True:
                print(f'{self.__str__()} wrote data to PE {self.dest_pe.relative_id} Read Buffer')

            # Debug - if ANY component has been accessed, set to True
            self.source_pe.accessed = True


class PEGB(Bus):

    def __init__(self, pes, bitwidth=DEFAULT_BUS_BITWIDTH, debug=False):
        super(PEGB, self).__init__(bitwidth)

        # List of PE's connected to this PEGB
        self.pes = pes

        self.new_data_written = False

        self.debug = debug

    def __str__(self):
        s = f'[PEGB] data: {self.data}, bitwidth {self.bitwidth}'
        return s

    def read(self):
        pass

    def write(self, data):
        assert isinstance(data, Data)
        self.data = data
        self.new_data_written = True

    def write_to_pe_read_buffer(self):
        """
        PEGB -> PEGB destination Read Buffer
        """
        if self.new_data_written:
            pe_id_relative = self.data.rel_pe_id
            dest_pe = self.pes[pe_id_relative]
            read_buffer = dest_pe.pegb_read_buffer
            data = self.data.value
            read_buffer.push(data, dest_pe.absolute_id, dest_pe.relative_id)
            # TODO This could be a source of bug...maybe not b/c only one thing can read this at a time...
            self.new_data_written = False
            if self.debug:
                print(f'[PEGB] wrote data {data} to PE {dest_pe.relative_id} Read Buffer')
            dest_pe.accessed = True
            # TODO only for debug purposes - remove return statements after done
            return True
        return False


    def read_from_pe_write_buffer(self, pe_id_relative):
        """
        PEGB Write Buffer -> PEGB
        """
        source_pe = self.pes[pe_id_relative]
        write_buffer = source_pe.pegb_write_buffer
        data = write_buffer.pop()
        if self.debug:
            print(f'[PEGB] read data {data} from PE {source_pe.relative_id} Write Buffer')
        return data


class PUNB(Bus):
    """
    PU Neighbor Bus. Connects PU N to PU N + 1. Essentially identical to PENB.
    """

    def __init__(self, source_pu, dest_pu, bitwidth=DEFAULT_BUS_BITWIDTH, debug=False):
        super(PUNB, self).__init__(bitwidth)
        self.source_pu = source_pu
        self.dest_pu = dest_pu

        self.source_head_pe = self.source_pu.head_pe
        self.dest_head_pe = self.dest_pu.head_pe

        # Flag to be set if new data has been written to this bus. Set to False
        # once data is read (because the data is now in PUNB Read Buffer).
        # Note: Since buses only have one value at a time, once data is read from
        # it, it won't be read again.
        self.new_data_written = False

        self.debug = debug


    def __str__(self):
        s = f'[PUNB] PU {self.source_pu.id} -> PU {self.dest_pu.id}, ' + \
            f'data: {self.data}, ' + \
            f'bitwidth: {self.bitwidth}'
        return s

    def read(self):
        self.new_data_written = False
        return self.data

    def write(self, data):
        """
        Write data value to this bus. PU N (source PU) sets the value this PU Neighbor Bus
        """
        assert isinstance(data, Data)
        self.data = data
        self.new_data_written = True

    def write_to_pu_read_buffer(self):
        """
        Write the data in this bus to the Read Buffer of destination PU.
        """
        if self.new_data_written:
            read_buffer = self.dest_head_pe.punb_read_buffer
            read_buffer.push(self.data, self.dest_head_pe.absolute_id, self.dest_head_pe.relative_id)
            self.new_data_written = False
            if self.debug == True:
                print(f'{self.__str__()} wrote data to PU {self.dest_pu.id} Read Buffer')

            # Debug - if ANY component has been accessed, set to True
            self.source_head_pe.accessed = True
            return True
        return False


class PUGB(Bus):

    def __init__(self, pus, bitwidth=DEFAULT_BUS_BITWIDTH, debug=False):
        super(PUGB, self).__init__(bitwidth)

        # List of PU's connected to this PUGB
        self.pus = pus

        self.new_data_written = False

        self.debug = debug

    def __str__(self):
        s = f'[PUGB] data: {self.data} ' + \
            f'(bitwidth {self.bitwidth})'
        return s

    def read(self):
        pass

    def write(self, data: Data):
        assert isinstance(data, Data)
        self.data = data
        self.new_data_written = True

    def get_pu_id(self, absolute_id):
        for pu in self.pus:
            for pe in pu.pes:
                if pe.absolute_id == absolute_id:
                    return pu.id

    def write_to_pu_read_buffer(self):
        """
        PUGB -> destination PUGB Read Buffer
        """
        if self.new_data_written:
            pu_id = self.get_pu_id(self.data.abs_pe_id)
            dest_pu = self.pus[pu_id]
            head_pe = dest_pu.head_pe
            read_buffer = head_pe.pugb_read_buffer
            read_buffer.push(self.data, head_pe.absolute_id, head_pe.relative_id)
            self.new_data_written = False
            if self.debug:
                print(f'PUGB wrote {self.data} to PU {dest_pu.id} Read Buffer')
            head_pe.accessed = True
            # TODO only for debug purposes - remove return statements after done
            return True
        return False


    def read_from_pu_write_buffer(self, pu_id):
        """
        Soruce PUGB Write Buffer -> PUGB
        """
        source_pu = self.pus[pu_id]
        head_pe = source_pu.head_pe
        write_buffer = head_pe.pugb_write_buffer
        data = write_buffer.pop()
        if self.debug:
            print(f'PUGB read {data} from PU {source_pu.id} Write Buffer')
        return data


if __name__ == '__main__':
    # pegb = PEGB()
    # print(pegb)

    penb = PENB()
    penb.set_data(3)
    print(penb)
