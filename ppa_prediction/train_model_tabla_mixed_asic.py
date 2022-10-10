'''
Helper function to train and test perticular model or models
'''
import os
from datetime import datetime

cwd = os.getcwd()
from rtml_ppa import *
from rtml_iccad_helper1 import *
os.chdir(cwd)
import h2o
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.estimators.deeplearning import H2ODeepLearningEstimator

pid = os.getpid()
tnow = datetime.now()
prefix = f'{tnow.month}-{tnow.day}-{tnow.year}_{tnow.hour}-{tnow.minute}-{tnow.second}'
model_type = 'mixed'
design_name = 'tabla'
platform_name = 'asic'

# Load TABLA asic data
train_df, test_df = load_tabla_mixed_asic()
backend_features, system_features, metrics = load_tabla_asic_features_and_metrics()
train, tests = startH2o(train_df, [test_df], name = 'RTML'+str(pid))
metrics = metrics[2:]
test = tests[0]

seed = 42
folds = 5
assignment_type = "Modulo"
ids = ['performance', 'power', 'runtime', 'energy']
saved_models = f'XX_PATH_XX/saved_model/{model_type}_{platform_name}/{design_name}/{prefix}'

if not os.path.isfile(saved_models) and not os.path.isdir(saved_models):
    os.mkdir(saved_models)

for metric in metrics:
    i = find_metric(metrics, metric)
    if i < 2:
        x = backend_features
    else:
        x = system_features
    y = metric
    grid_id = ids[i]

    print(f'Current Metric: {metric}')
    ## Train DL
    print("Starting DL")
    dl_grids = randTrainDL(train, x, y, dl_param_list = dl_param_list,\
                    search_criteria = search_criteria_dl, idx = 'dl_'+grid_id)
    test_df = PredictAndPlot(dl_grids, test_df, test, y, no = 1)

    sorted_dl_grids = dl_grids.get_grid(sort_by='rmse', decreasing=False)
    best_dl = sorted_dl_grids.models[0]
    best_dl1 = sorted_dl_grids.models[1]
    dl_model_path = h2o.save_model(model=best_dl, path=saved_models, force = True)
    # Train XGB
    print("Starting XGB")
    xgb_grids = randTrainGradientBoost(train, x, y,\
                                    xgb_param_list = xgb_param_list,\
                                    search_criteria = search_criteria,\
                                    idx = 'xgb_'+grid_id)
    test_df = PredictAndPlot(xgb_grids, test_df, test, y, no = 2)

    sorted_xgb_grids = xgb_grids.get_grid(sort_by='rmse', decreasing=False)
    best_xgb = sorted_xgb_grids.models[0]
    best_xgb1 = sorted_xgb_grids.models[1]
    xgb_model_path = h2o.save_model(model=best_xgb, path=saved_models, force = True)
    
    # Train RF
    print("Starting RF")
    mtries = [i+1 for i in range(len(x))]
    rf_param_list['mtries'] = mtries
    rf_grids = randTrainRF(train, x, y, rf_param_list = rf_param_list,\
                        search_criteria = search_criteria, idx = 'rf_'+grid_id)
    test_df = PredictAndPlot(rf_grids, test_df, test, y, no = 3)
    sorted_rf_grids = rf_grids.get_grid(sort_by='rmse', decreasing=False)
    best_rf = sorted_rf_grids.models[0]
    best_rf1 = sorted_rf_grids.models[1]
    rf_model_path = h2o.save_model(model=best_rf, path=saved_models, force = True)

    # Train Ensemble
    print("Starting Ensemble")
    sorted_rf_grids = rf_grids.get_grid(sort_by='rmse', decreasing=False)
    best_rf = sorted_rf_grids.models[0]
    best_rf1 = sorted_rf_grids.models[1]
    
    best_models = [best_xgb, best_dl, best_rf, best_xgb1, best_rf1, best_dl1]
    ensemble = H2OStackedEnsembleEstimator( model_id = 'my_ensemble'+grid_id,
                            base_models = best_models,
                            metalearner_algorithm='glm')
    ensemble.train(x = x, y = y, training_frame = train )
    test_df = plotPredictionUpdateTest(ensemble, 4, y, test_df, test)
    test_df.reset_index(drop = True)
    test_df.to_csv(f'Output_CSV/{design_name}_{platform_name}_{model_type}_test_{prefix}.1.csv')

test_df.to_csv(f'Output_CSV/{design_name}_{platform_name}_{model_type}_test_{prefix}.csv')
