'''
Helper function to generate plots, MAPE and two stage evaluation
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
import seaborn as sn
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

vector_axiline = {
    2: 500000,
    3: 500000,
    1: 500000,
    4: 500000
}

benchmarkMap = {'LINEAR':1, 'LOGISTIC':2, 'SVM':3, 'RECO':4}

search_criteria = {
        'strategy': "RandomDiscrete",
        'stopping_metric':'RMSE',
        'seed': 42,
        'max_runtime_secs': 300
    }

param_list = {
            'max_depth':[i for i in range(3, 50,1)],
            'ntrees': [i for i in range(30,300,5)],
        }


def mlReport(y1, y2, isPrint = 1, detail = 0):
    df = pd.DataFrame()
    if type(y1).__name__ == "Series":
        df['act'] = y1.to_numpy()
    elif type(y1).__name__ == "ndarray":
        df['act'] = y1.reshape(-1)
    else:
        df['act'] = np.array(y1).reshape(-1)

    if type(y2).__name__ == "Series":
        df['predict'] = y2.to_numpy()
    elif type(y2).__name__ == "ndarray":
        df['predict'] = y2.reshape(-1)
    else:
        df['predict'] = np.array(y2).reshape(-1)

    df['p_error'] = 100*np.abs(df['predict'] - df['act'])/df['act']
    error = np.mean(df['p_error'].to_numpy())
    std_error = np.std(df['p_error'].to_numpy())
    max_error = np.max(df['p_error'].to_numpy())

    if isPrint == 1:
        print(f'Percentage Error: {error}')
        print(f'STD Percentage Error: {std_error}')
        print(f'Max Percentage Error: {max_error}')


    if detail == 1:
        return error, std_error, df
    elif detail == 2:
        return error, max_error
    elif detail == 3:
        return error, max_error, std_error
    return error, std_error

def PlotVio(df, metric, predicted_metric,
            parameter = 'target_clock_frequency(GHz)',
            marker = 'r*',ax = None, label=None):
    df_local = df.copy()
    df_local['abs_error'] = abs(df_local[metric] - df_local[predicted_metric])*\
                                100/df_local[metric]
  
    if ax == None:
        fig = plt.figure(figsize=(8,5), dpi = 200)
        gspec = plt.GridSpec(ncols=1, nrows = 1, figure=fig)
        ax = fig.add_subplot(gspec[0,0])
        ax.set_ylabel("% Error")
        ax.set_title(f"Error plot for {metric} preidction")
    ax.set_xlabel(parameter)
    ax.plot(df_local[parameter],df_local['abs_error'],f"{marker}", \
            label=f"{label}")
    return

def plotResult(df, metric, predicted_metric,
                xmetric = 'target_clock_frequency(GHz)', ax = 0):
    if ax == 0:
        fig = plt.figure(figsize=(10,5), dpi = 300)
        gspec = plt.GridSpec(ncols=1, nrows = 1, figure=fig)
        ax = fig.add_subplot(gspec[0,0])
  
    ax.plot(df[xmetric],df[predicted_metric],'r.', label = "Prediction")
    ax.plot(df[xmetric],df[metric],'b.', label = 'actual')
    plt.legend()

    for i in range(len(df)):
        x_temp = []
        y_temp = []
        x_temp.append(df.iloc[i][xmetric])
        y_temp.append(df.iloc[i][predicted_metric])
        x_temp.append(df.iloc[i][xmetric])
        y_temp.append(df.iloc[i][metric])
        ax.plot(x_temp, y_temp, "-k", linewidth = 1)
  
    ax.set_xlabel(f"{xmetric}")
    ax.set_ylabel(f"{metric}")
    test_error, max_error, test_std_error = mlReport(df[metric], 
                                            df[predicted_metric], detail = 3)
    ax.set_title(f"Test MAPE:{round(test_error,2)}% " 
                f"Max MAPE:{round(max_error,2)}%"
                f"STD APE:{round(test_std_error,2)}")
    return

def highVio(prediction, df, metric, parameter, th = 5):
    df_local = df.copy()
    p1 = parameter.copy()
    if type(p1).__name__ == 'set':
        p1 = list(p1)
    p1.append(metric)
    p1.append('Predicted')
    p1.append('abs_error')
    df_local['Predicted'] = prediction
    df_local['abs_error'] = abs(df_local[metric] - df_local['Predicted'])*\
                                100/df_local[metric]
    display(df_local[df_local['abs_error']>=th][p1].sort_values(
                            by=['abs_error'], ascending = False))
    return

def HybridErrorPlot(updated_test_df, no, r, metric, r1 = None, r2 = None,
                    parameter = 'target_clock_frequency(GHz)'):

    metric1 = 'effective_clock_frequency(GHz)'
    metric11= 'target_clock_frequency(GHz)'

    predicted_metric1 = 'predict'+str(no)+'_'+metric1
    predicted_metric = 'predict'+str(no)+'_'+metric

    plot_df = updated_test_df[(updated_test_df[predicted_metric1] <= \
                             (1+r)*updated_test_df[metric11]) & \
                            (updated_test_df[predicted_metric1] >= \
                            (1-r)*updated_test_df[metric11])]

    predic = ((updated_test_df[predicted_metric1] <= \
                (1+r)*updated_test_df[metric11]) & 
                (updated_test_df[predicted_metric1] >= \
                (1-r)*updated_test_df[metric11]))

    act = ((updated_test_df[metric1] <= (1+r)*updated_test_df[metric11]) & 
            (updated_test_df[metric1] >= (1-r)*updated_test_df[metric11]))

    #print(predic, act)
    cfMatrix = confusion_matrix(act,predic)
    cmsum = sum(sum(cfMatrix))
    cfMatrix = (cfMatrix/cmsum)
    print(cfMatrix)
    print(f"Accuracy: {cfMatrix[0][0] + cfMatrix[1][1]}")
    #sn.heatmap(cfMatrix, annot = True, fmt = ".2%" , annot_kws = {"size": 16})
    #plt.draw()
    plot_df = updated_test_df[(updated_test_df[metric1] <= 
                             (1+r)*updated_test_df[metric11]) & 
                             (updated_test_df[metric1] >= 
                             (1-r)*updated_test_df[metric11])]

    fig = plt.figure(figsize = (8, 5), dpi = 200)
    gspec = plt.GridSpec(ncols = 1, nrows = 1, figure = fig)
    ax = fig.add_subplot(gspec[0, 0])
    mlReport(plot_df[metric], plot_df[predicted_metric])
    PlotVio(updated_test_df, metric, predicted_metric, marker = 'r*', ax = ax,
            parameter = parameter)
    PlotVio(plot_df, metric, predicted_metric, marker = 'b.', ax = ax,
            parameter = parameter)
    plt.draw()
    #ax = fig.add_subplot(gspec[0, 0])
    ax.set_ylabel("% Error")
    ax.set_title(f"Error plot for {metric} preidction")
    return

def axilineSpnrDataGen(pnr_rpt, dc_rpt):
    pnr_df = pd.read_csv(pnr_rpt)
    pnr_df.rename(columns={'tcp' : 'target_clock_period(ps)', 
                'tcf' : 'target_clock_frequency(GHz)',
                'effective_clock_period' : 'effective_clock_period(ps)',
                'effective_clock_frequency' : 'effective_clock_frequency(GHz)',
                'switching_power' : 'switching_power(mW)',
                'leakage_power' : 'leakage_power(mW)',
                'internal_power' : 'internal_power(mW)',
                'total_power' : 'total_power(mW)'
                }, inplace = True
            )
    dc_df = pd.read_csv(dc_rpt)
    dc_df.rename(
        columns={
            'tcp' : 'target_clock_period(ps)', 
            'tcf' : 'target_clock_frequency(GHz)',
            'dc_effective_cp' : 'dc_effective_clock_period(ps)',
            'dc_switching_power' : 'dc_switching_power(mW)',
            'dc_leakage_power' : 'dc_leakage_power(mW)',
            'dc_internal_power' : 'dc_internal_power(mW)',
            'dc_total_power' : 'dc_total_power(mW)'
            }, inplace= True
        )
    dc_df['dc_effective_clock_frequency(GHz)'] = \
            1e3/dc_df['dc_effective_clock_period(ps)']
    dc_df = dc_df.drop(['target_clock_frequency(GHz)'], axis = 1)
    dc_df = updateAxilineDf(dc_df, pref = 'dc')
    pnr_df = updateAxilineDf(pnr_df)
    spnr_df = pnr_df.merge(dc_df, how = 'inner', on = ['benchmark', 'run_type', 
                        'size', 'num_cycle', 'num_unit', 'bit_width', 
                        'target_clock_period(ps)'])
    return spnr_df

def findClockCycle (numCycle, vectorCount):
    z = numCycle*2 + 1
    return z + (z - numCycle + 1)*(vectorCount - 1)

def updateAxilineDf(df, pref = None):
    l = df.shape[0]
    if pref == None:
        df['energy(uJ)'] = [0.0]*l
        df['runtime(ms)'] = [0.0]*l
        df['vector_count'] = [0]*l
        df['benchmark_no'] = [0]*l
    else:
        df[pref+'_energy(uJ)'] = [0.0]*l
        df[pref+'_runtime(ms)'] = [0.0]*l

    for i in range(l):
        design = df.loc[i,'benchmark']
        numCycle = df.loc[i,'num_cycle']
        clockCycle = findClockCycle(numCycle, vector_axiline[benchmarkMap[design]])
        if pref == None:
            eff_cp = df.loc[i,'effective_clock_period(ps)']
            power = df.loc[i,'total_power(mW)']
            runtime = eff_cp*clockCycle*1e-9
            energy = power*runtime
            df.at[i, 'vector_count'] = vector_axiline[benchmarkMap[design]]
            df.at[i, 'energy(uJ)'] = energy
            df.at[i, 'runtime(ms)'] = runtime
            df.at[i, 'benchmark_no'] = benchmarkMap[design]
        else:
            eff_cp = df.loc[i,pref+'_effective_clock_period(ps)']
            power = df.loc[i,pref+'_total_power(mW)']
            runtime = eff_cp*clockCycle*1e-9
            energy = power*runtime
            df.at[i, pref+'_energy(uJ)'] = energy
            df.at[i, pref+'_runtime(ms)'] = runtime
    return df

def plotPredictionUpdateTest(h2o_model, no, metric, test_df, test_h2o_df, ax=0):
    predicted_metric = f'predict{no}_{metric}'
    prediction = h2o_model.predict(test_h2o_df)
    test_df[predicted_metric] = prediction.as_data_frame()
    mlReport(test_df[metric], test_df[predicted_metric])
    PlotVio(test_df, metric, predicted_metric)
    plotResult(test_df, metric, predicted_metric, ax = ax)
    return test_df

def startH2o(train_df, test_dfs = [], name = 'RTML', nthreads = 16):
    h2o.init(nthreads = nthreads, max_mem_size = '60G', name = name)
    train = h2o.H2OFrame(train_df)
    tests = []
    for test_df in test_dfs:
        tests.append(h2o.H2OFrame(test_df))
    return train, tests

def loadAxilineSVM():
    pnr_rpt = '/home/zf4_projects/RTML/sakundu/PNR/ICCAD22/rpt_dir/'\
                'axiline_svm_train.csv'
    dc_rpt = '/home/zf4_projects/RTML/sakundu/PNR/ICCAD22/rpt_dir/'\
                'axiline_train_dc.csv'
    spnr_df = axilineSpnrDataGen(pnr_rpt, dc_rpt)

    config_df = spnr_df[['benchmark', 'run_type', 'size', 'num_cycle',\
                        'num_unit']]
    config_df = config_df.drop_duplicates().reset_index(drop = True)
    train_config = config_df.iloc[0:8]
    test_config1 = config_df.iloc[8:10]
    test_config2 = config_df.iloc[10:12]
    test_config3 = config_df.iloc[12:14]

    train_df = spnr_df.copy()
    train_df = shuffle(train_df, random_state = 42)
    train_df = train_df.merge(train_config, how = 'inner', on = ['benchmark',
                'run_type', 'size', 'num_cycle', 'num_unit'])
    test_df1 = spnr_df.copy()
    test_df1 = test_df1.merge(test_config1, how = 'inner', on = ['benchmark',
                'run_type', 'size', 'num_cycle', 'num_unit'])
    test_df2 = spnr_df.copy()
    test_df2 = test_df2.merge(test_config2, how = 'inner', on = ['benchmark',
                'run_type', 'size', 'num_cycle', 'num_unit'])
    test_df3 = spnr_df.copy()
    test_df3 = test_df3.merge(test_config3, how = 'inner', on = ['benchmark',
                'run_type', 'size', 'num_cycle', 'num_unit'])
    #print(train_df.shape, test_df1.shape, test_df2.shape, test_df3.shape)

    test_dfs = [test_df1, test_df2, test_df3]
    train, tests = startH2o(train_df, test_dfs)
    return train, tests, train_df, test_dfs

def h2oShutdown():
    h2o.cluster().shutdown()
    h2o.shutdown(prompt = False)
    return

search_criteria = {
        'strategy': "RandomDiscrete",
        'stopping_metric':'RMSE',
        'seed': 42,
        'max_runtime_secs': 300
    }

xgb_param_list = {
    'max_depth':[i for i in range(3, 100,1)],
    'ntrees': [i for i in range(30,500,5)],
    }

def randTrainXGBoos(train, x, y, xgb_param_list = xgb_param_list,
                    search_criteria = search_criteria, idx = 'xgb_grid',
                    assignment_type = 'Modulo', folds = 5, seed = 42):

    xgb_grid = H2OGridSearch(model = H2OXGBoostEstimator(
                                    fold_assignment = assignment_type,
                                    nfolds = folds, seed = seed,
                                    keep_cross_validation_predictions = True),
                                grid_id = idx,
                                search_criteria = search_criteria,
                                hyper_params = xgb_param_list)

    xgb_grid.train(x = x, y = y,
            training_frame = train,
            seed = seed, parallelism = 8)

    return xgb_grid

def randTrainGradientBoost(train, x, y, xgb_param_list = xgb_param_list,
                    search_criteria = search_criteria, idx = 'gdb_grid',
                    assignment_type = 'Modulo', folds = 5, seed = 42):
                    
    gdb_grid = H2OGridSearch(model = H2OGradientBoostingEstimator(
                                    fold_assignment = assignment_type,
                                    nfolds = folds, seed = seed,
                                    keep_cross_validation_predictions = True),
                                grid_id = idx,
                                search_criteria = search_criteria,
                                hyper_params = xgb_param_list)

    gdb_grid.train(x = x, y = y,
            training_frame = train,
            seed = seed, parallelism = 8)

    return gdb_grid


rf_param_list = {
                    'max_depth':[i for i in range(5, 150, 1)],
                    'ntrees': [i for i in range(30, 600, 5)],
                }

def randTrainRF(train, x, y, rf_param_list = rf_param_list, 
                search_criteria = search_criteria, idx = 'rf_grid',
                assignment_type = 'Modulo', folds = 5, seed = 42):
    rf_grid = H2OGridSearch(model = H2ORandomForestEstimator(
                                    fold_assignment = assignment_type,
                                    nfolds = folds, seed = seed,
                                    keep_cross_validation_predictions = True),
                            grid_id = idx,
                            hyper_params = rf_param_list,
                            search_criteria = search_criteria,
                        )

    rf_grid.train(x = x, y = y,
            training_frame = train,
            seed = seed, parallelism = 8)
    return rf_grid

dl_layers = []
for d in range(3,7,1):
    for nc in range(5, 101, 5):
        ll = [nc for i in range(d)]
        dl_layers.append(ll)

# search_criteria_dl = {
#     'strategy': "RandomDiscrete",
#     'stopping_metric':'MSE',
#     'seed': 42,
#     'max_runtime_secs': 600
# }
# 
# dl_param_list = {'activation': ['Tanh', 'Rectifier', 'Maxout'],
#                 'hidden': dl_layers,
#                 'epochs' : [300, 200],
#                 'rate': [0.01, 0.005, 0.001],
#                 }

search_criteria_dl = {
    'strategy': "RandomDiscrete",
    'stopping_metric':'RMSE',
    'stopping_tolerance':0.0001,
    'stopping_rounds': 10,
    'seed': 42,
    'max_runtime_secs': 600
}

dl_param_list = {'activation': ['Tanh', 'Rectifier', 'Maxout'],
                'hidden': dl_layers,
                'rate': [0.01, 0.005, 0.001, 0.0001],
                'rate_decay': [1, 0.9, 0.8],
                'rate_annealing': [1e-6, 1e-5, 1e-4, 1e-3],
                'momentum_start': [0.5, 0.4, 0.6, 0.3, 0],
                'momentum_ramp' : [1e6, 1e4, 1e2],
                'momentum_stable' : [0.99, 0.9, 0.8, 0]
                }

# def randTrainDL(train, x, y, dl_param_list = dl_param_list,
#                 search_criteria = search_criteria_dl, idx = 'dl_grid',
#                 assignment_type = 'Modulo', folds = 5, seed = 42):

#     dl_grid_rand = H2OGridSearch(model = H2ODeepLearningEstimator(
#                                 fold_assignment = assignment_type,
#                                 nfolds = folds, seed = seed,
#                                 keep_cross_validation_predictions = True),
#                                 grid_id = idx,
#                                 hyper_params = dl_param_list,
#                                 search_criteria = search_criteria,
#                                 )

#     dl_grid_rand.train(x = x, y = y,
#                 training_frame = train,
#                 seed = seed, parallelism = 16)
#     return dl_grid_rand

def randTrainDL(train, x, y, dl_param_list = dl_param_list,
                stopping_tolerance = 0.00001,
                search_criteria = search_criteria_dl, idx = 'dl_grid',
                assignment_type = 'Modulo', folds = 5, seed = 42):

    dl_grid_rand = H2OGridSearch(model = H2ODeepLearningEstimator(
                                fold_assignment = assignment_type,
                                nfolds = folds, seed = seed,
                                epochs = 10000,
                                reproducible = True, stopping_metric = 'RMSE',
                                stopping_rounds = 20, adaptive_rate = False,
                                stopping_tolerance = stopping_tolerance,
                                keep_cross_validation_predictions = True),
                                grid_id = idx,
                                hyper_params = dl_param_list,
                                search_criteria = search_criteria,
                                )

    dl_grid_rand.train(x = x, y = y,
                training_frame = train,
                seed = seed, parallelism = 8)
    return dl_grid_rand

def PredictAndPlot(grids, test_df, test, y, no = 1):
    sorted_grids = grids.get_grid(sort_by='rmse', decreasing=False)
    best_grid = sorted_grids.models[0]
    return plotPredictionUpdateTest(best_grid, no, y, test_df, test)


def sortModelGrids(grid_list, n = 3, k = 5):
    models = []
    rmse_values = []
    for grid in grid_list:
        sorted_grid = grid.get_grid(sort_by='rmse', decreasing=False)
        for i in range(n):
            models.append(sorted_grid.models[i])
            rmse_values.append(sorted_grid.models[i].rmse())
    sorted_indexes = np.argsort(rmse_values)
    base_models = []
    for i in range(k):
        base_models.append(models[sorted_indexes[i]])
    return base_models
