from _collections import deque


class PEGBArbiter(object):

    def __init__(self, pegb, debug=False):
        # PE Global Bus object
        self.pegb = pegb

        self.pes_per_pu = len(self.pegb.pes)

        self.cycle = 0
        self.debug = debug


    def __str__(self):
        s = 'PEGB Bus Arbiter'
        return s

    def find_pes_with_write_buffer(self):
        """
        Go through every write buffer in PEGB and find all the non-empty ones.
        """
        pe_ids_relative = []
        for pe in self.pegb.pes:
            pegb_write_buffer = pe.pegb_write_buffer
            if pegb_write_buffer.peek_fifo():
                pe_ids_relative.append(pe.relative_id)
            # if pegb_write_buffer.new_data_written_count > 0:
            #     pe_ids_relative.append(pe.relative_id)
        return pe_ids_relative

    def set_priority(self):
        """
        Set priority values for each PE in the given cycle.
        """
        priority_list = deque(range(self.pes_per_pu))
        priority_list.rotate(self.cycle)
        return priority_list

    def find_highest_priority_pe_id(self, pe_ids, priority_list):
        relative_pe_ids = list(map(lambda x : x % self.pes_per_pu, pe_ids))
        rel_pe_priorities = list(map(lambda x : priority_list[x], relative_pe_ids))
        index = rel_pe_priorities.index(min(rel_pe_priorities))
        return pe_ids[index]

    def run_one_cycle(self):
        # if self.debug:
        #     print(f'PEGB Bus Arbiter: {self.pegb.data} ([value, PE ID]) in PEGB')

        # PEGB -> PEGB Read Buffer
        # TODO self.accessed only for debug purposes...
        self.accessed = False
        self.accessed = self.pegb.write_to_pe_read_buffer()

        pe_ids_relative = self.find_pes_with_write_buffer()
        priority_list = self.set_priority()
        # if self.debug:
        #     print(f"PE IDs: {pe_ids_relative}, Priority list: {priority_list}")

        # If there's no PE's with any values on the PEGB Write Buffer, don't do anything
        if len(pe_ids_relative) == 0:
            self.cycle += 1
            return

        pe_id = self.find_highest_priority_pe_id(pe_ids_relative, priority_list)
        # if self.debug:
        #     print(f"Highest priority PE ID: {pe_id}\n")

        # PEGB Write Buffer -> PEGB
        data = self.pegb.read_from_pe_write_buffer(pe_id)
        self.accessed = True
        self.pegb.write(data)
        # if self.debug:
        #     print(f'{self.pegb.data}')

        self.cycle += 1



class PUGBArbiter(object):

    def __init__(self, pugb, debug=False):
        # PU Global Bus object
        self.pugb = pugb

        self.num_pus = len(self.pugb.pus)

        self.cycle = 0
        self.debug = debug

    def __str__(self):
        s = 'PUGB Bus Arbiter'
        return s

    def find_pus_with_write_buffer(self):
        """
        Return a list of PU ID's that has data in its PUGB Write buffer. Go
        through every write buffer in PEGB and find all the non-empty ones.
        """
        pu_ids = []
        for pu in self.pugb.pus:
            pugb_write_buffer = pu.pugb_write_buffer
            if pugb_write_buffer.peek_fifo():
                pu_ids.append(pu.id)
            # if pugb_write_buffer.new_data_written_count > 0:
            #     pu_ids.append(pu.id)
        return pu_ids

    def set_priority(self):
        """
        Set priority values for each PU in the given cycle.
        """
        priority_list = deque(range(self.num_pus))
        priority_list.rotate(self.cycle)
        return priority_list

    def find_highest_priority_pu_id(self, pu_ids, priority_list):
        relative_pu_ids = list(map(lambda x : x % self.num_pus, pu_ids))
        rel_pu_priorities = list(map(lambda x : priority_list[x], relative_pu_ids))
        index = rel_pu_priorities.index(min(rel_pu_priorities))
        return pu_ids[index]

    def run_one_cycle(self):
        # if self.debug:
        #     print(f'PUGB Bus Arbiter: {self.pugb.data} ([value, PU ID]) in PUGB (0 means bus is empty)')

        # PUGB -> PUGB Read Buffer
        # TODO self.accessed only for debug purposes...
        self.accessed = False
        self.accessed = self.pugb.write_to_pu_read_buffer()

        pu_ids = self.find_pus_with_write_buffer()
        priority_list = self.set_priority()

        # If there's no PU's with any values on the PUGB Write Buffer, don't do anything
        if len(pu_ids) == 0:
            self.cycle += 1
            return

        pu_id = self.find_highest_priority_pu_id(pu_ids, priority_list)
        # if self.debug:
        #     print(f'Highest priority PU ID: {pu_id}')
        self.accessed = True
        # PUGB Write Buffer -> PUGB
        data = self.pugb.read_from_pu_write_buffer(pu_id)
        self.pugb.write(data)
        # if self.debug:
        #     print(f"PUGB Arbiter: Wrote {self.pugb.data} to PUGB")

        self.cycle += 1
        return self.accessed
