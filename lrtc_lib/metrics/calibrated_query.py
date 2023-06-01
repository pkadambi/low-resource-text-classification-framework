from sklearnex import patch_sklearn
patch_sklearn()
from data_utils.fashionmnist_binary_load import *
from trainer_class import *
from trainers import *

nseed = 50
ncentroid = 9
budget_max = 1000
query_step = 10
npoints = int((budget_max - nseed) / query_step) - 2
kvalue = 3
n_MC = 30
xtrain, ytrain, xtest, ytest = sample_ankle_sneaker_data()


posterior_method = 'knn'
classifier = 'mlp'

def run_twoclass(xtrain, ytrain, xtest, ytest, posterior_method, kvalue, n_MC, nseed, query_step,
                 budget_max, ncentroid, savedir):
    unifs = []
    uncertainty = []
    uncertainty_calib_pred = []

    if 'logistic' in posterior_method:
        posterior_fn = logistic_posterior_estimate
        savedir = os.path.join(savedir, posterior_method+'_posteriorest')

    elif 'knn' in posterior_method :
        posterior_fn = knn_estimate_posterior
        savedir = os.path.join(savedir, posterior_method+f'{kvalue}nn_posteriorest')

    else:
        raise Exception('Error: argument posterior_method should be "logistic" or "knn"')


    for ii in range(n_MC):
        #--------------------
        dataset = ActiveCalibrationDataset(xdata=xtrain, ydata=ytrain, xtest=xtest, ytest=ytest,
                                           nseed=nseed, nclusters=ncentroid)

        trainer = BinaryMLPCalibratedQueryTrainer(dataset=dataset, kvalue=kvalue, acquisition_fn=active_uncertainty_acqusition_fn,
                                    nclasses=2, savefname='uncertainty', alldata_stat=True,
                                    savedir=savedir, posterior_fn=posterior_fn, query_calib=False)
        uncertainty_ = trainer.train(budget_max=budget_max, query_step=query_step)
        uncertainty.append(uncertainty_)

        #--------------------
        dataset = ActiveCalibrationDataset(xdata=xtrain, ydata=ytrain, xtest=xtest, ytest=ytest,
                                           nseed=nseed, nclusters=ncentroid)

        trainer = BinaryMLPCalibratedQueryTrainer(dataset=dataset, kvalue=kvalue, acquisition_fn=active_uncertainty_acqusition_fn,
                                    nclasses=2, savefname='knnactive_allstat', alldata_stat=True,
                                    savedir=savedir, posterior_fn=posterior_fn, query_calib=True)
        uncertainty_ = trainer.train(budget_max=budget_max, query_step=query_step)
        uncertainty_calib_pred.append(uncertainty_)

        # -------------------
        dataset = ActiveCalibrationDataset(xdata=xtrain, ydata=ytrain, xtest=xtest, ytest=ytest,
                                           nseed=nseed, nclusters=ncentroid)

        trainer = BinaryMLPCalibratedQueryTrainer(dataset=dataset, kvalue=kvalue, acquisition_fn=unif_acqusition_fn,
                             nclasses=2, savefname='unif', posterior_fn=posterior_fn,
                                    savedir=savedir)
        unif_rslt1 = trainer.train(budget_max=budget_max, query_step=query_step)
        unifs.append(unif_rslt1)

    dct_list = [unifs, uncertainty, uncertainty_calib_pred]

    final_results_dct = os.path.join(savedir, 'final_results')
    final_results_dct = os.path.join(final_results_dct, posterior_method)
    os.makedirs(final_results_dct, exist_ok=True)
    pkl.dump({'rslts_final': dct_list}, open(final_results_dct + 'results.pkl', 'wb'))

    dct_list = [results_collate(dct, npoints) for dct in dct_list]
    sample_sizes = np.linspace(nseed, budget_max, npoints)
    return dct_list, sample_sizes

savepath = './results/fashion_anklesneaker_' + classifier + f'clf_{kvalue}nn'

dct_list, sample_sizes = run_twoclass(xtrain, ytrain, xtest, ytest, posterior_method=posterior_method,
                                      n_MC=n_MC, kvalue=kvalue, ncentroid=ncentroid, nseed=nseed,
                                      budget_max=budget_max, query_step=query_step,
                                      savedir=savepath)


# plot_results(dct_list, sample_sizes,
#              methodnames=['unif', 'KUB_tshirtsneaker', 'KUB_allstat_tshirtsneaker',
#                           'KUB_UC_tshirtsneaker', 'KUB_UC_allstat_tshirtsneaker',
#                           'Uncert', 'Uncert_UC'])

plot_results(dct_list, sample_sizes,
             methodnames=['unif', 'uncertainty', 'calibrated_uncertainty'])
plt.show()
print()



