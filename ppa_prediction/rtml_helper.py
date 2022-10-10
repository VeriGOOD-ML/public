'''
Helper function to generate plots, MAPE and two stage evaluation
'''
import pandas as pd
import numpy as np

def load_tabla_mixed_asic():
    mixed_train_csv = 'XX_PATH_XX/TABLA/'\
                    'TABLA_train_mixed.csv'
    mixed_test_csv = 'XX_PATH_XX/TABLA/'\
                    'TABLA_test_mixed.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)

    ## Dropping outliers
    test_df = test_df.reset_index()
    train_df = train_df.reset_index()

    return train_df, test_df

def load_tabla_unseen_asic():
    train_df, test_df = load_tabla_mixed_asic()
    tabla_df = pd.concat([train_df, test_df]).reset_index(drop = True)
    config_dir = 'XX_PATH_XX/'
    train_config_df = pd.read_csv(config_dir+'tabla_train_config.csv')
    test_config_df = pd.read_csv(config_dir+'tabla_test_config.csv')
    train_df = tabla_df.merge(train_config_df, how = 'inner',
                on = ['design_name'])
    test_df = tabla_df.merge(test_config_df, how = 'inner',
                on = ['design_name'])

    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    return train_df, test_df

def load_tabla_mixed_kintex_fpga():
    mixed_train_csv = 'XX_PATH_XX/'\
                    'Kintex_mixed_train.csv'
    mixed_test_csv = 'XX_PATH_XX/'\
                    'Kintex_mixed_test.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)
    train_df = train_df[train_df['target_clock_frequency(GHz)'] > 0.051].\
                reset_index(drop = True)
    test_df = test_df[test_df['target_clock_frequency(GHz)'] > 0.051].\
                reset_index(drop = True)

    return train_df, test_df

def load_tabla_mixed_virtex_fpga():
    mixed_train_csv = 'XX_PATH_XX/'\
                    'virtex_7_mixed_train.csv'
    mixed_test_csv = 'XX_PATH_XX/'\
                    'virtex_7_mixed_test.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)

    train_df = train_df[train_df['target_clock_frequency(GHz)'] > 0.051].\
                reset_index(drop = True)
    test_df = test_df[test_df['target_clock_frequency(GHz)'] > 0.051].\
                reset_index(drop = True)

    return train_df, test_df

def load_tabla_unseen_virtex_fpga():
    train_df, test_df = load_tabla_mixed_virtex_fpga()
    tabla_df = pd.concat([train_df, test_df]).reset_index(drop = True)
    config_dir = 'XX_PATH_XX/'
    train_config_df = pd.read_csv(config_dir+'tabla_train_config_fpga.csv')
    test_config_df = pd.read_csv(config_dir+'tabla_test_config_fpga.csv')
    train_df = tabla_df.merge(train_config_df, how = 'inner',
                on = ['Design'])
    test_df = tabla_df.merge(test_config_df, how = 'inner',
                on = ['Design'])

    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    return train_df, test_df

def load_tabla_unseen_kintex_fpga():
    train_df, test_df = load_tabla_mixed_kintex_fpga()
    tabla_df = pd.concat([train_df, test_df]).reset_index(drop = True)
    config_dir = 'XX_PATH_XX/'
    train_config_df = pd.read_csv(config_dir+'tabla_train_config_fpga.csv')
    test_config_df = pd.read_csv(config_dir+'tabla_test_config_fpga.csv')
    train_df = tabla_df.merge(train_config_df, how = 'inner',
                on = ['Design'])
    test_df = tabla_df.merge(test_config_df, how = 'inner',
                on = ['Design'])

    train_df = train_df.reset_index(drop = True)
    test_df = test_df.reset_index(drop = True)

    return train_df, test_df

def load_axiline_mixed_asic():
    mixed_train_csv = 'XX_PATH_XX/Output_CSV/'\
                    'axiline_mixed_train.csv'
    mixed_test_csv = 'XX_PATH_XX/Output_CSV/'\
                    'axiline_mixed_test.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)
    return train_df, test_df

def load_axiline_unseen_asic():
    mixed_train_csv = 'XX_PATH_XX/Output_CSV/'\
                    'axiline_unseen_train.csv'
    mixed_test_csv = 'XX_PATH_XX/Output_CSV/'\
                    'axiline_unseen_test.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)

    return train_df, test_df

def load_axiline_mixed_dc():
    mixed_train_csv = 'XX_PATH_XX/Output_CSV/'\
                        'axiline_spnr_train_mixed_asic.csv'
    mixed_test_csv = 'XX_PATH_XX/Output_CSV/'\
                        'axiline_spnr_test_mixed_asic.csv'

    train_df = pd.read_csv(mixed_train_csv)
    test_df = pd.read_csv(mixed_test_csv)
    return train_df, test_df


def load_axiline_asic_features_and_metrics():
    axiline_parameters = ['size', 'num_cycle', 'num_unit', 'bit_width', \
                        'benchmark_no', 'target_clock_frequency(GHz)']
    axiline_metrics = ['effective_clock_frequency(GHz)', 'total_power(mW)', \
                        'runtime(ms)', 'energy(uJ)']
    return axiline_parameters, axiline_metrics

def load_tabla_fpga_features_and_metrics():
    tabla_backend_parameters = ['num_pu', 'num_pe', 'bit_width', 'internal_bit_width',
                    'benchmark', 'target_clock_frequency(GHz)']

    tabla_system_parameters = ['vector_count','num_pu', 'num_pe', 'bit_width',
                    'internal_bit_width', 'benchmark',
                    'target_clock_frequency(GHz)']
    tabla_fpga_metrics = ['effective_clock_frequency(GHz)', 'total_power(mW)',\
                        'runtime(s)', 'energy(J)']
    
    return tabla_backend_parameters, tabla_system_parameters, tabla_fpga_metrics

def load_tabla_asic_features_and_metrics():
    tabla_backend_parameters = ['num_pu', 'num_pe', 'bit_width', 'internal_bit_width',
                    'benchmark', 'target_clock_frequency(GHz)']

    tabla_system_parameters = ['vector_count','num_pu', 'num_pe', 'bit_width',
                    'internal_bit_width', 'benchmark',
                    'target_clock_frequency(GHz)']
    tabla_fpga_metrics = ['effective_clock_frequency(GHz)', 'total_power(mW)',\
                        'runtime(ms)', 'energy(uJ)']
    
    return tabla_backend_parameters, tabla_system_parameters, tabla_fpga_metrics

def find_metric(metrics, metric):
    for i in range(len(metrics)):
        if metrics[i] == metric:
            return i
    return -1

def load_vta_unseen_asic():
    train_csv = 'XX_PATH_XX/VTA_RPT/vta_unseen_train.csv'
    test_csv = 'XX_PATH_XX/VTA_RPT/vta_unseen_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df

def load_vta_mixed_asic():
    train_csv = 'XX_PATH_XX/VTA_RPT/vta_mixed_train.csv'
    test_csv = 'XX_PATH_XX/VTA_RPT/vta_mixed_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df

def load_vta_asic_features_and_metrics():
    backend_parameter = ['PE', 'WBUF', 'IBUF', 'OBUFF', 'BW',
                        'target_clock_frequency(GHz)']
    system_parameter = ['PE', 'WBUF', 'IBUF', 'OBUFF', 'BW',
                        'target_clock_frequency(GHz)']
    metrics = ['effective_clock_frequency(GHz)', 'total_power(mW)', 'runtime(s)',
                'energy(J)']
    return backend_parameter, system_parameter, metrics

def load_genesys_unseen_asic():
    train_csv = 'XX_PATH_XX/GeneSys_RPT/genesys_unseen_tarin.csv'
    test_csv = 'XX_PATH_XX/GeneSys_RPT/genesys_unseen_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df

def load_genesys_mixed_asic():
    train_csv = 'XX_PATH_XX/GeneSys_RPT/genesys_mixed_tarin.csv'
    test_csv = 'XX_PATH_XX/GeneSys_RPT/genesys_mixed_test.csv'

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    return train_df, test_df

def load_genesys_asic_features_and_metrics():
    backend_parameter = ['ACT_DATA_WIDTH', 'ARRAY_M', 'IBUF_AXI_DATA_WIDTH',
                    'SIMD_VMEM_CAPACITY_BITS', 'WBUF_CAPACITY_BITS',
                    'target_clock_frequency(GHz)']
    system_parameter = ['ACT_DATA_WIDTH', 'ARRAY_M', 'IBUF_AXI_DATA_WIDTH',
                    'SIMD_VMEM_CAPACITY_BITS', 'WBUF_CAPACITY_BITS',
                    'target_clock_frequency(GHz)']
    metrics = ['effective_clock_frequency(GHz)', 'total_power(mW).1', 'runtime(s)', 'energy(J)']
    return backend_parameter, system_parameter, metrics
