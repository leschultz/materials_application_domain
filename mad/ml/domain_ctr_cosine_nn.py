from mad.functions import parallel, llh, set_llh, poly, extract_embeddings
from mad.ml import distances, distances_maha_ctr, distances_cosine_ctr, distances_cosine_nn
from mad.functions import chunck
from sklearn.base import clone
from sklearn.neural_network import MLPClassifier


import pandas as pd
import numpy as np
import random
import dill
import glob
import os

import torch
from mad.representation.get_repr_maha import train_representation_maha
from mad.representation.get_repr_cosine import train_representation_cosine

import warnings
warnings.filterwarnings('ignore')


def define_labels(dataframe, std, bins, quantile = 0.5 ):
    total_size = dataframe.shape[0]
    bin_res = list(chunck(dataframe['absres'].values,bins))
    bin_stdcal = list(chunck(dataframe['stdcal'].values, bins))
    bin_idx = list(chunck(dataframe['idx'].values, bins))
    rmse = np.array([(np.ma.sum(i**2)/len(i))**0.5 for i in bin_res])
    avg_stdcal = np.array([np.ma.mean(i) for i in bin_stdcal])
    err_in_err = abs( rmse - avg_stdcal)/std
    cut = pd.Series(err_in_err).quantile([quantile]).values[0]

    # assert False, f"err_in_err:{err_in_err}\nCut:{cut}"

    labels = [ 0 for i in range(total_size) ] # default label 0 for OOD
    for i in range(bins):
        if err_in_err[i] < cut:
            # set label = 1 if in-domain
            for idx in bin_idx[i]:
                labels[idx] = 1
    return labels

class uq_func_model:

    def __init__(self, params, uq_func):
        self.params = params
        self.uq_func = uq_func

    def train(self, std, y, y_pred):

        params = set_llh(
                         std,
                         y,
                         y_pred,
                         self.params,
                         self.uq_func
                         )

        self.params = params

    def predict(self, std):
        return self.uq_func(self.params, std)


class dist_func_model:

    def train(self, X, y=None):
        self.dist_func = lambda x: distances.distance(X, x, y)

    def predict(self, X):
        return self.dist_func(X)

class dist_func_model_cosine_nn:

    def train(self, X, model, y=None):
        self.dist_func = lambda x: distances_cosine_nn.distance(X, x, model = model)

    def predict(self, X):
        return self.dist_func(X)

class dist_func_model_cosine_ctr:

    def train(self, X, y=None):
        self.dist_func = lambda x: distances_cosine_ctr.distance(X, x, y)

    def predict(self, X):
        return self.dist_func(X)

class builder:
    '''
    Class to use the ingredients of splits to build a model and assessment.
    '''

    def __init__(
                 self,
                 pipe,
                 X,
                 y,
                 d,
                 splitter,
                 save,
                 seed=1,
                 uq_func=poly,
                 uq_coeffs_start=[0.0, 1.0],
                 dataset_name = None,
                 joint_domain = False
                 ):
        '''
        inputs:
            pipe = The machine learning pipeline.
            X = The features.
            y = The target variable.
            d = The domain for each case.
            splitter = The test set splitter.
            splitters = The splitting oject to create 2 layers.
            save = The directory to save splits.
            seed = The seed option for reproducibility.
        '''

        # Setting seed for reproducibility
        np.random.seed(seed)
        np.random.RandomState(seed)
        random.seed(seed)

        self.pipe = pipe
        self.X = X
        self.y = y
        self.d = d
        self.splitter = splitter
        self.uq_func = uq_func
        self.uq_coeffs_start = uq_coeffs_start

        # Output directory creation
        self.save = save
        self.dataset_name = dataset_name
        # if True, then use case_by_case err in err (stdcal) and binned err in err together.
        # if False, only use binned error in error
        self.joint_domain = joint_domain

    def assess_domain(self):
        '''
        Asses the model through nested CV with a domain layer.
        '''

        o = np.array(range(self.X.shape[0]))  # Tracking cases.

        # Setup saving directory.
        save = os.path.join(self.save, 'splits')
        os.makedirs(save, exist_ok=True)

        # Make all of the train and test splits.
        test_count = 0  # In domain count
        splits = []
        for i in self.splitter.split(self.X, self.y, self.d):

            tr_index = np.array(i[0])  # The train.
            te_index = np.array(i[1])  # The test.

            tr_te = (
                     tr_index,
                     te_index,
                     test_count,
                     )

            splits.append(tr_te)

            test_count += 1  # Increment in domain count

        # Do nested CV
        parallel(
                 self.nestedcv,
                 splits,
                 X=self.X,
                 y=self.y,
                 d=self.d,
                 pipe=self.pipe,
                 save=save,
                 uq_func=self.uq_func,
                 uq_coeffs_start=self.uq_coeffs_start,
                 dataset_name = self.dataset_name,
                 joint_domain = self.joint_domain
                 )

    def nestedcv(
                 self,
                 indexes,
                 X,
                 y,
                 d,
                 pipe,
                 save,
                 uq_func,
                 uq_coeffs_start,
                 dataset_name, # for contrastive learning setup
                 joint_domain,
                 ):
        '''
        A class for nesetd cross validation.

        inputs:
            indexes = The in domain test and training indexes.
            X = The feature set.
            y = The target variable.
            d = The class.
            pipe = The machine learning pipe.
            save = The saving directory.
            uq_coeffs_start = The starting coefficients for UQ polynomial.

        outputs:
            df = The dataframe for all evaluation.
        '''

        # Split indexes and spit count
        tr, te, test_count = indexes

        X_train, X_test = X[tr], X[te]
        y_train, y_test = y[tr], y[te]
        d_train, d_test = d[tr], d[te]

        # Fit the model on training data in domain.
        self.pipe.fit(X_train, y_train)

        # Grab model critical information for assessment
        pipe_best = pipe.best_estimator_
        pipe_best_scaler = pipe_best.named_steps['scaler']
        pipe_best_select = pipe_best.named_steps['select']
        pipe_best_model = pipe_best.named_steps['model']

        if 'manifold' in pipe_best.named_steps:
            pipe_best_manifold = pipe_best.named_steps['manifold']

        # Grab model specific details
        model_type = pipe_best_model.__class__.__name__

        # Feature transformations
        X_train_trans = pipe_best_scaler.transform(X_train)
        X_test_trans = pipe_best_scaler.transform(X_test)

        if 'manifold' in pipe_best.named_steps:
            X_train_trans = pipe_best_manifold.transform(X_train)
            X_test_trans = pipe_best_manifold.transform(X_test)

        # Feature selection
        X_train_select = pipe_best_select.transform(X_train_trans)
        X_test_select = pipe_best_select.transform(X_test_trans)

        # Setup distance model
        dists = dist_func_model()
        dists.train(X_train_select, y_train)

        # Calculate distances after feature transformations from ML workflow.
        df_te = dists.predict(X_test_select)

        n_features = X_test_select.shape[-1]

        # If model is ensemble regressor (need to update varialbe name)
        ensemble_methods = [
                            'RandomForestRegressor',
                            'BaggingRegressor',
                            'GradientBoostingRegressor',
                            'GaussianProcessRegressor'
                            ]

        if model_type in ensemble_methods:

            # Train and test on inner CV
            std_cv = []
            d_cv = []
            y_cv = []
            y_cv_pred = []
            y_cv_indx = []
            df_tr = []

            # train feature collector by gather X_test in nested_cg
            train_collect = []
            train_targets = []
            for train_index, test_index in pipe.cv.split(
                                                         X_train_select,
                                                         y_train,
                                                         d_train
                                                         ):

                model = clone(pipe_best_model)

                X_cv_train = X_train_select[train_index]
                X_cv_test = X_train_select[test_index]
                # gather "test" in our train collection in order
                train_collect.append(X_cv_test)

                y_cv_train = y_train[train_index]
                y_cv_test = y_train[test_index]
                # gather "test" in our train collection in order
                train_targets.append(y_cv_test )

                model.fit(X_cv_train, y_cv_train)

                if model_type == 'GaussianProcessRegressor':
                    _, std = model.predict(X_cv_test, return_std=True)
                else:
                    std = []
                    for i in model.estimators_:
                        if model_type == 'GradientBoostingRegressor':
                            i = i[0]
                        std.append(i.predict(X_cv_test))

                    std = np.std(std, axis=0)

                dists_cv = dist_func_model()
                dists_cv.train(X_cv_train, y_cv_train)

                std_cv = np.append(std_cv, std)
                d_cv = np.append(d_cv, d_train[test_index])
                y_cv = np.append(y_cv, y_cv_test)
                y_cv_pred = np.append(y_cv_pred, model.predict(X_cv_test))
                y_cv_indx = np.append(y_cv_indx, tr[test_index])
                df_tr.append(pd.DataFrame(dists_cv.predict(X_cv_test)))

            df_tr = pd.concat(df_tr)

            # Calibration
            uq_func = uq_func_model(uq_coeffs_start, uq_func)
            uq_func.train(std_cv, y_cv, y_cv_pred)

            # Nested prediction for left out data
            y_test_pred = pipe_best.predict(X_test)

            # Ensemble predictions with correct feature set
            if model_type == 'GaussianProcessRegressor':
                _, std_test = pipe_best_model.predict(
                                                      X_test_select,
                                                      return_std=True
                                                      )

            else:
                pipe_estimators = pipe_best_model.estimators_
                std_test = []
                for i in pipe_estimators:

                    if model_type == 'GradientBoostingRegressor':
                        i = i[0]

                    std_test.append(i.predict(X_test_select))

                std_test = np.std(std_test, axis=0)

            stdcal_cv = uq_func.predict(std_cv)
            stdcal_test = uq_func.predict(std_test)

            # Grab standard deviations.
            df_tr['std'] = std_cv
            df_te['std'] = std_test

            # Grab calibrated standard deviations.
            df_tr['stdcal'] = stdcal_cv
            df_te['stdcal'] = stdcal_test

            # Contrastive learning
            train_ctr = np.concatenate(train_collect)
            train_ctr_target = np.concatenate(train_targets)
            std = np.std(train_ctr_target)
            train_ctr = torch.tensor(train_ctr )
            train_size = train_ctr.shape[0]
            test_ctr = torch.tensor(X_test_select)

            df_temp = pd.DataFrame()
            df_temp["stdcal"] = stdcal_cv
            df_temp['absres'] = abs( y_cv_pred - y_cv )
            df_temp['idx'] = [ i for i in range(train_size)] # index use for indexing sample
            df_temp = df_temp.sort_values(by=['stdcal', 'absres'])

            # some "hyper" param need to refine later
            bin_num = 8 # better to be a even number

            labels = define_labels( dataframe = df_temp,
                                    std = std,
                                    bins = bin_num,
                                    quantile = 0.95 )
            cases_by_case_err = stdcal_cv
            cut = pd.Series(cases_by_case_err ).quantile([0.7]).values[0]
            if joint_domain:
                # consider 2 definition of domain together
                labels2 = [1 if err < cut else 0 for err in cases_by_case_err ]
                assert len(labels) == len(labels2)
                labels_joint = [ 1 if labels2[i] + labels[i] == 2 else 0 for i in range(len(labels2))]
                labels = labels_joint

            assert dataset_name != None
            repr_model = train_representation_cosine(train_ctr, labels, dataset_name)

            repr_tr = extract_embeddings(train_ctr,repr_model) # np.array type
            repr_te = extract_embeddings(test_ctr,repr_model)
            # BECAREFUL df_id now is just a dictionary!

            # group train_id and train_ood
            id_mask = np.array(labels) == 1
            ood_mask = ~id_mask

            repr_tr_id = repr_tr[id_mask]
            repr_tr_ood = repr_tr[ood_mask ]

            dists_ctr = dist_func_model_cosine_ctr()
            # dits1 = train_id VS. All train
            dists_ctr.train(repr_tr_id)
            tr_distance = dists_ctr.predict(repr_tr) # in domain training distances
            # dits2 = train_id VS. test
            te_distance = dists_ctr.predict(repr_te)
            #
            # insert modified distances
            for key in tr_distance:
                df_tr[key] = tr_distance[key]
            for key in te_distance:
                df_te[key] = te_distance[key]


            # use representation to generate softmax score as metrics
            dists_nn_cosine = dist_func_model_cosine_nn()

            if dataset_name == "large":
                max_iter = 1000
                batch_size = 15

            elif dataset_name == "small":
                max_iter = 300
                batch_size = 10

            elif dataset_name == "middle":
                max_iter = 500
                batch_size = 15

            hidden_layer_sizes=(100,50)
            if repr_tr.shape[1] > 100:
                hidden_layer_sizes=(200,200)

            model = MLPClassifier(  random_state=1,
                                    batch_size = batch_size,
                                    max_iter=max_iter,
                                    hidden_layer_sizes = hidden_layer_sizes)
            model.fit(repr_tr, labels)

            dists_nn_cosine.train(X = repr_tr, model = model ) # 1 is in-domain, 0 for OOD
            # dist score is the probability to be OOD
            tr_distance_nn = dists_nn_cosine.predict(repr_tr) # in domain training distances
            te_distance_nn = dists_nn_cosine.predict(repr_te)
            # assert False f""
            for key in tr_distance_nn:
                df_tr[key] = tr_distance_nn[key]
            for key in te_distance_nn:
                df_te[key] = te_distance_nn[key]


        else:
            raise Exception('Only ensemble models supported.')

        # Assign domain.
        df_tr['in_domain'] = ['tr']*std_cv.shape[0]
        df_te['in_domain'] = ['te']*X_test.shape[0]

        # Grab indexes of tests.
        df_tr['index'] = y_cv_indx
        df_te['index'] = te

        # Grab the domain of tests.
        df_tr['domain'] = d_cv
        df_te['domain'] = d_test

        # Grab the true target variables of test.
        df_tr['y'] = y_cv
        df_te['y'] = y_test

        # Grab the predictions of tests.
        df_tr['y_pred'] = y_cv_pred
        df_te['y_pred'] = y_test_pred

        # Calculate the negative log likelihoods
        df_tr['nllh'] = -llh(
                             std_cv,
                             y_cv-y_cv_pred,
                             uq_func.params,
                             uq_func.uq_func
                             )
        df_te['nllh'] = -llh(
                             std_test,
                             y_test-y_test_pred,
                             uq_func.params,
                             uq_func.uq_func
                             )

        df_tr = pd.DataFrame(df_tr)
        df_te = pd.DataFrame(df_te)

        df = pd.concat([df_tr, df_te])

        # Assign values that should be the same
        df['test_count'] = test_count

        dfname = 'split_{}.csv'.format(test_count)
        modelname = 'model_{}.joblib'.format(test_count)
        uqname = 'uqfunc_{}.joblib'.format(test_count)
        distname = 'distfunc_{}.joblib'.format(test_count)

        dfname = os.path.join(save, dfname)
        modelname = os.path.join(save, modelname)
        uqname = os.path.join(save, uqname)
        distname = os.path.join(save, distname)

        df.to_csv(dfname, index=False)
        dill.dump(pipe, open(modelname, 'wb'))
        dill.dump(uq_func, open(uqname, 'wb'))
        dill.dump(dists, open(distname, 'wb'))

    def aggregate(self):
        '''
        Gather all data from domain analysis.
        '''

        files = glob.glob(self.save+'/splits/split_*')

        df = parallel(pd.read_csv, files)
        df = pd.concat(df)

        name = os.path.join(self.save, 'aggregate')
        os.makedirs(name, exist_ok=True)
        name = os.path.join(name, 'data.csv')
        df.to_csv(name, index=False)

        return df
