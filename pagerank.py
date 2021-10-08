import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import operator
import argparse


def init_hyperlink_matrix(graph_in):

	nodes_list = list(graph_in.nodes)
	nodes_num = len(nodes_list)
	H = np.zeros([nodes_num, nodes_num])

	for n, neighbours in graph_in.adj.items():
		neighbours_num = len(list(neighbours.items()))
		for neighbour in neighbours.items():
			H[neighbour[0], n] = 1 / neighbours_num

	return H


def rayleigh_quotient(eigen_vector, H):
	eigenvalue = eigen_vector.T @ H @ eigen_vector / (eigen_vector.T @ eigen_vector)
	return eigenvalue


def power_method(H, iterations):

	# Random importance vector
	I = np.random.rand(H.shape[0])
	I = I / np.linalg.norm(I)
	eigenval_old = rayleigh_quotient(I, H)
	initial_I = I
	print(eigenval_old)

	print('\nComputing power method...\n')
	for i in range(iterations):

		# New impotance vector
		I = np.dot(H, I)
		I = I / np.linalg.norm(I)

		# Difference between new and old eigen values (for debugging purpose)
		# eigenval = rayleigh_quotient(I, H)
		# eigenval = np.linalg.norm(I)
		# error = abs(eigenval - eigenval_old)
		# print(f'error = {error},\t eigen_val = {eigenval}')
		# eigenval_old = eigenval
	
	print('\nFinished.\n')
	return I, initial_I


def main():

	parser = argparse.ArgumentParser(description='PageRank')

	# PageRank parameters
	parser.add_argument('--alpha', default=0.85, type=float, help='dumping factor')
	parser.add_argument('--iterations', default=100, type=int, help='iterations for computing importance vector')

	# Graph Parameters
	parser.add_argument('--nodes', default=25, type=int, help='graph nodes number')
	parser.add_argument('--edge_prob', default=0.125, type=float, help='Edge probability for creating the random graph')
	parser.add_argument('--seed', default=9999, type=int, help='Rand seed')
	parser.add_argument('--plots', default=False, action='store_true', help='enable_plots (enable when nodes number <= 50)')

	args = parser.parse_args()

	# Create random graph
	directed = True
	sample_graph = nx.fast_gnp_random_graph(args.nodes, args.edge_prob, args.seed, directed)
	
	# Create hyperlink matrix
	H = init_hyperlink_matrix(sample_graph)

	# Transform H to stochastic matrix
	col_sum = np.sum(H, axis=0)
	for s in range(col_sum.shape[0]):
		if col_sum[s] != 0.:
			continue
		H[:, s] = 1 / float(H.shape[0])

	# Apply damping factor
	GG = args.alpha * H + (1 - args.alpha) / H.shape[0]
	
	# Compute importance vector I through the power method
	I, initial_I = power_method(GG, args.iterations)

	# NetworkX PageRank (for comparison)
	pr = nx.pagerank(sample_graph, alpha=args.alpha, max_iter=args.iterations)

	# From dict to list
	I_nx = [pr[key] for key in pr]

	I = I.tolist()
	initial_I = initial_I.tolist()
	pr_custom = {}
	pr_custom_initial = {}
	for i in range(len(I)):
		pr_custom[i] = I[i]
		pr_custom_initial[i] = initial_I[i]

	# Plots
	# Compare the custom implementation with the algorithm of the NetworkX library
	if args.plots:
		plt.rcParams.update({'font.size': 23, 'font.weight': 'bold'})

		fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
		ax1.set_title("Initial")
		nx.draw(sample_graph, ax=ax1, cmap=plt.get_cmap('OrRd'), node_color=initial_I, with_labels=True, font_weight='bold',
				pos=nx.spring_layout(sample_graph, seed=99))
		ax2.set_title("Custom implementation")
		nx.draw(sample_graph, ax=ax2, cmap=plt.get_cmap('OrRd'), node_color=I, with_labels=True, font_weight='bold',
				pos=nx.spring_layout(sample_graph, seed=99))
		ax3.set_title("NetworkX")
		nx.draw(sample_graph, ax=ax3, cmap=plt.get_cmap('OrRd'), node_color=I_nx, with_labels=True, font_weight='bold',
				pos=nx.spring_layout(sample_graph, seed=99))
		plt.show()


if __name__ == '__main__':
	main()
