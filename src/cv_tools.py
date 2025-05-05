import pandas as pd
import numpy as np
from tools import save_df_to_file
from os.path import isfile

from probinet.evaluation.link_prediction import compute_link_prediction_AUC
from scipy.special import factorial
from sklearn.metrics import log_loss

from probinet.input.loader import build_adjacency_from_file


'''
---------------------------------------------------------------
				CV
---------------------------------------------------------------
'''

def get_df_train_test(df: pd.DataFrame,
					  data,
					  cv_mask: dict,
					  fold: int=0,
					  outdir: str = '../../../data/outdir/wto/cv/',
					  filename0: str = 'wto_aob_fold',
					  source: str = 'reporter_name',
					  target: str = 'partner_name',
					  weight: str = 'weight',
					  undirected: bool = True,
					  force_dense: bool = True,
					  binary: bool = True,
					  ):
	'''
	Get a dataframe similar to the original df, splitted into train and test
	This will then be given in input to probinet for loading the data_cv
	'''

	filename_train = f'{filename0}{fold}_train.csv'
	filename_test = f'{filename0}{fold}_test.csv'
	cond1 = isfile(f"{outdir}{filename_train}")
	cond2 = isfile(f"{outdir}{filename_test}")
	if (cond1 & cond2) == False: # if both files are not yet present, then build df and save them

		nodeId2Label = {i: n for i, n in enumerate(data.nodes)}

		# Test set
		subs = np.where(cv_mask[fold] > 0)[1:]
		nodes1 = [nodeId2Label[i] for i in subs[0]]
		nodes2 = [nodeId2Label[i] for i in subs[1]]
		ws = data.adjacency_tensor[cv_mask[fold]].astype(int)
		df_test = pd.DataFrame({source: nodes1, target: nodes2, weight: ws})

		# Train set
		df_train = df.merge(df_test[df_test[weight] > 0], how='left', indicator=True).query('_merge == "left_only"').drop(
			'_merge', axis=1)
		df_train = pd.concat([df_train, df_test[df_test[weight] == 0]], axis=0)
		assert len(df) == len(df_train[df_train[weight] > 0]) + len(df_test[df_test[weight] > 0])

		save_df_to_file(df_train, filename=filename_train, outdir=outdir)
		save_df_to_file(df_test, filename=filename_test, outdir=outdir)

	data_cv = build_adjacency_from_file(
		f"{outdir}{filename_train}",
		ego=source,
		alter=target,
		sep=",",
		undirected=undirected,
		force_dense=force_dense,
		binary=binary,
		header=0,
	)
	return data_cv

def compute_mean_lambda0_em(u,v,w):
    if w.ndim == 3:
        return np.einsum('ik,jq,akq->aij',u,v,w)
    else:
        if w.shape[0] == w.shape[1]:
            Y = np.zeros((1,u.shape[0],v.shape[0]))
            Y[0,:] = np.einsum('ik,jq,kq->ij',u,v,w)
            return Y
        else:
            return np.einsum('ik,jk,ak->aij',u,v,w)

def get_loglikelihood(
	B: np.ndarray,
	u: np.ndarray,
	v: np.ndarray,
	w: np.ndarray,
	y_pred: np.ndarray = None,
	mask: np.ndarray = None,
	EPS: float = 1e-12
) -> float:
	"""
	Compute the log-likelihood for the network structure.

	Parameters
	----------
	B : np.ndarray
		Graph adjacency tensor.
	u : np.ndarray
		Membership matrix (out-degree).
	v : np.ndarray
		Membership matrix (in-degree).
	w : np.ndarray
		Affinity tensor.
	mask : Optional[np.ndarray]
		Mask for selecting a subset in the adjacency tensor.

	Returns
	-------
	float
		Log-likelihood value for the network structure.
	"""
	if y_pred is not None:
		if mask is None:
			return (B * np.log(y_pred + EPS)).sum() - y_pred.sum() -(np.log(factorial(B.astype(int)))).sum()
		else:
			M = y_pred[mask > 0]
			logM = np.zeros(M.shape)
			logM[M > 0] = np.log(M[M > 0])
			return (B[mask > 0] * logM).sum() - M.sum() - (np.log(factorial(B[mask > 0].astype(int)))).sum()


	if mask is None:
		# Compute the expected adjacency tensor
		M = compute_mean_lambda0_em(u,v,w)
		logM = np.zeros(M.shape)
		logM[M > 0] = np.log(M[M > 0])
		return (B * logM).sum() - M.sum() -(np.log(factorial(B.astype(int)))).sum()

	# Compute the expected adjacency tensor for the masked elements
	lambda_poisson = compute_mean_lambda0_em(u,v,w)
	M = lambda_poisson[mask > 0]
	logM = np.zeros(M.shape)
	logM[M > 0] = np.log(M[M > 0])
	return (B[mask > 0] * logM).sum() - M.sum() - (np.log(factorial(B[mask > 0].astype(int)))).sum()


def get_prediction_results(data, params_cv: dict,mask_test: np.ndarray, Y_pred: dict=None, fold: int = 0)-> pd.DataFrame:
	'''
	Collect inferred parameters and compute predictive performance results.
	Organize in a pd.DataFrame
	'''
	if Y_pred is None:
		Y_pred = {algo: compute_mean_lambda0_em(v[0], v[1], v[2]) for algo, v in params_cv.items()}

	auc_train = [np.round(compute_link_prediction_AUC(data.adjacency_tensor,y_pred, mask=np.logical_not(mask_test[fold])),3) for a,y_pred in Y_pred.items()]
	auc_test = [np.round(compute_link_prediction_AUC(data.adjacency_tensor,y_pred, mask=mask_test[fold]),3) for a,y_pred in Y_pred.items()]
	df_auc = pd.DataFrame({'algo':[a for a in Y_pred.keys()],'auc_train':auc_train,'auc_test':auc_test})

	logL_test = [np.round(get_loglikelihood(data.adjacency_tensor,v[0],v[1],v[2],mask = mask_test[fold],y_pred=Y_pred[a]),3) for a,v in params_cv.items()]
	logL_train = [np.round(get_loglikelihood(data.adjacency_tensor,v[0],v[1],v[2],mask = np.logical_not(mask_test[fold]),y_pred=Y_pred[a]),3) for a,v in params_cv.items()]
	bce_test = [np.round(log_loss(data.adjacency_tensor[mask_test[fold]],y_pred[mask_test[fold]]),3) for a,y_pred in Y_pred.items()]
	bce_train = [np.round(log_loss(data.adjacency_tensor[np.logical_not(mask_test[fold])],y_pred[np.logical_not(mask_test[fold])]),3) for a,y_pred in Y_pred.items()]

	df_holl = pd.DataFrame({'algo':[a for a in params_cv.keys()],'logL_train':logL_train,'logL_test':logL_test,
							'bce_train':bce_train,'bce_test':bce_test
						   })

	df_auc = df_auc.merge(df_holl,on='algo')

	if fold is None: fold = -1
	df_auc.loc[:,'fold'] = fold
	# Sort columns
	cols = ['fold','algo']
	for t in ['test','train']:
		for metric in ['auc','logL','bce']:
			cols.append(f"{metric}_{t}")
	df_auc = df_auc[cols]

	return df_auc


'''
---------------------------------------------------------------
	TOOLS TO BUILD MASKS
---------------------------------------------------------------
'''


def shuffle_indices_symmetric(shape, seed: int = 10):
	'''
	To extract a symmetric mask containing (A_ij,A_ji)
	'''
	L = shape[0]
	N = shape[-1]
	n_samples = int(N * (N - 1) * 0.5)  # upper triangle excluding diagonal
	indices = [np.arange(n_samples) for l in range(L)]
	rng = np.random.RandomState(seed)
	for l in range(L): rng.shuffle(indices[l])
	return indices


def extract_mask_symmetric_kfold(indices, N, NFold: int = 5):
	'''
	Symmetric mask: contains pairs (i,j) and (j,i) --> for undirected networks
	KFold : no train/test sets intersect across the K folds
	'''
	L = len(indices)
	mask = {f: np.zeros((L, N, N), dtype=bool) for f in range(NFold)}
	for fold in range(NFold):
		for l in range(L):
			n_samples = len(indices[l])
			test = indices[l][fold * (n_samples // NFold):(fold + 1) * (n_samples // NFold)]
			# train = list(set(indices).difference(set(test)))
			mask0 = np.zeros((n_samples), dtype=bool)
			mask0[test] = 1
			mask[fold][l][np.triu_indices(N, k=1)] = mask0
			mask[fold][l] = mask[fold][l] + mask[fold][l].T
	return mask


def extract_mask(shape, out_mask=False, outfolder: str = '../../../data/output/tests/cv/', outfile=None,
				 seed: int = 10, NFold: int = 5):
	indices = shuffle_indices_symmetric(shape, seed=seed)
	mask = extract_mask_symmetric_kfold(indices, shape[-1], NFold=NFold)

	if out_mask:  # output the masks into files
		for fold in mask.keys():
			outmask = outfolder + outfile + '_' + str(fold)
			np.savez_compressed(outmask + '.npz', mask=np.where(mask[fold] > 0))
			# To load: mask = np.load('mask_f0.npz'), e.g. print(np.array_equal(maskG, mask['maskG']))
			print('Masks saved in:', outmask)

	return mask

