from codelets.adl.graph import ArchitectureNode


def sa_mvmul_template(hag: ArchitectureNode):
    instructions = []
    # buffers = [('weight', 'WBUF'), ('data','IBUF'), ('bias', 'BBUF'), ('out', 'OBUF')]
    # for b in buffers:
    #     instructions += buffer_sa_template_compute(*b, hag)
    #     if b[1] == 'OBUF':
    #         instructions += sa_buffer_template_compute(*b, hag)

    return instructions