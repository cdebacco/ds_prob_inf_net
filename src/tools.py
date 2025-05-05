import pandas as pd
import numpy as np
import networkx as nx
from os import makedirs
from os.path import isdir

import matplotlib.pyplot as plt


def get_custom_node_positions(graph: nx.Graph)-> dict:

	dftmp = pd.DataFrame(index=graph.nodes(), columns=graph.nodes())
	for row, d in nx.shortest_path_length(graph):
		for col, dist in d.items():
			dftmp.loc[row,col] = dist

	# dftmp = dftmp.fillna(dftmp.max().max())
	dftmp = dftmp.infer_objects(copy=False)

	position = nx.kamada_kawai_layout(graph, dist=dftmp.to_dict())
	return position

def get_filename(filename: str, lecture_id: int = 1, file_extension: str = '.png') -> str:
	if lecture_id is None:
		return None
	return f"L{lecture_id}_{filename}{file_extension}"

def savefig(plt, outfile: str = None, outdir: str = 'tmp/', dpi: int = 150):

	if outfile is not None:
		if isdir(outdir) == False:
			makedirs(outdir)
			print(f"Creating output directory {outdir}")
			
		plt.savefig(f"{outdir}{outfile}", dpi=dpi, format=None, metadata=None,
					bbox_inches='tight', pad_inches=0.1,
					facecolor='auto', edgecolor='auto',
					backend=None
					)
		print(f"Figure saved in {outdir}{outfile}.")

def save_df_to_file(
	df: pd.DataFrame,
	outdir: str = None,
	filename: str = None,
	endfile: str = '.csv',
	index: bool = False
):
	if filename is not None:

		if isdir(outdir) == False:
			makedirs(outdir)

		if filename.endswith(endfile):
			filename = filename.replace(endfile,'')

		outfile = f"{outdir}{filename}{endfile}"

		print(f"DataFrame saved to {outfile}")

		if endfile.endswith('csv'):
			df.to_csv(outfile,index=index)
		elif endfile.endswith('xlsx'):
			df.to_excel(outfile,index=index)

def extract_node_order(membership_array: np.array) -> list:
	"""Return indexes to sort nodes based on their maximum membership.

	Parameters
	----------
	membership_array: membership matrix.

	Returns
	-------
	node_order: list with node indexes.
	"""
	node_order = []
	N, K = membership_array.shape
	k_max = np.argmax(membership_array, axis=1)
	for k in range(K):
		nodes_k = [i for i in range(N) if k_max[i] == k]
		nodes_k.sort()
		node_order.extend(nodes_k)
	assert len(node_order) == N
	return node_order

def fl(x, dp=2):
	return round(x, dp)

def extract_overlapping_membership(i, cm, U, threshold=0.001):
	groups = np.where(U[i] > threshold)[0]
	wedge_sizes = U[i][groups]
	wedge_colors = [cm(c) for c in groups]
	return wedge_sizes, wedge_colors

def normalize_nonzero_membership(U):
	den1 = U.sum(axis=1, keepdims=True)
	nzz = den1 == 0.
	den1[nzz] = 1.
	return U / den1


def CalculatePermuation(U_infer,U0):  
	"""
	Permuting the overlap matrix so that the groups from the two partitions correspond
	U0 has dimension NxK, reference memebership
	"""
	N,RANK=U0.shape
	M=np.dot(np.transpose(U_infer),U0)/float(N);   #  dim=RANKxRANK
	rows=np.zeros(RANK);
	columns=np.zeros(RANK);
	P=np.zeros((RANK,RANK));  # Permutation matrix
	for t in range(RANK):
	# Find the max element in the remaining submatrix,
	# the one with rows and columns removed from previous iterations
		max_entry=0.;c_index=1;r_index=1;
		for i in range(RANK):
			if columns[i]==0:
				for j in range(RANK):
					if rows[j]==0:
						if M[j,i]>max_entry:
							max_entry=M[j,i];
							c_index=i;
							r_index=j;
	 
		P[r_index,c_index]=1;
		columns[c_index]=1;
		rows[r_index]=1;

	return P

def from_louvain_to_u(louvain: list) -> np.ndarray:
	'''
	Builds one-hot encoded vector of dimension k=# groups
	'''
	N = sum([len(s) for s in louvain])
	K = len(louvain)
	u = np.zeros((N,K))
	for k, partition in enumerate(louvain):
		p = np.array(list(partition))
		u[p,k] = 1
	assert np.all(u.sum(axis=1)==1)
	return u

def from_hard_to_mixed(u_hard: np.ndarray) -> np.ndarray:
	'''
	Builds one-hot encoded vector of dimension k=# groups
	'''
	N = u_hard.shape[0]
	K = len(np.unique(u_hard))
	u = np.zeros((N,K))
	for i in np.arange(N):
		u[i,u_hard[i]] = 1
	assert np.all(u.sum(axis=1)==1)
	return u

def from_mixed_to_hard(u_mixed: np.ndarray) -> np.ndarray:
	'''
	Builds one-hot encoded vector of dimension k=# groups
	'''
	N, K = u_mixed.shape
	u = np.zeros((N,K))
	for i in np.arange(N):
		k = np.argmax(u_mixed[i])
		u[i,k] = 1
	assert np.all(u.sum(axis=1)==1)
	return u

def get_node_colors(colors, u):
	communities = np.argmax(u, axis=1)
	return [colors[c] for c in communities]

def get_logistic(x: float, beta: float = 1)-> float:
	'''
	Logistic function for score difference x
	'''
	den = 1 + np.exp(-2 * beta * x)
	return 1. / den

