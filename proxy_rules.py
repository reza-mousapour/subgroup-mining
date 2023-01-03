import numpy as np
from sklearn.cluster import AgglomerativeClustering as aglo
from scipy.spatial.distance import squareform
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import ete3
from collections import defaultdict
import random
import pandas as pd
import pickle
from math import log
import math

from optimize_clustering import find_best, create_clusters


#-------------------------------------------------------------------------
# ---- load data ----

def read_rules(file_name='rules.pkl'):
	"""
     Return exactly the pysubgroup output (a dataframe contain quality, subgroup, ...)
	"""
	return pd.read_pickle("pickle_data/" + file_name)


def read_id_set(file_name='id_set.pkl'):
	"""
        Return a dictionary {rule_ids:set_of_instances}
	"""
	with open('pickle_data/' + file_name, 'rb') as handle:
		return pickle.load(handle)


def read_ids(file_name='ids.pkl'):
	with open('pickle_data/' + file_name, 'rb') as handle:
		return pickle.load(handle)


#-------------------------------------------------------------------------
# ---- proxy by quality ----

def initial_proxies(n_clusters,clusters,info_rules):
	proxies = list()
	for clus in range(n_clusters):
		max_qual = 0
		max_i = -1
		for i in clusters[clus]:
			qual = info_rules.at[i, 'quality']
			if qual > max_qual:
				max_qual = qual
				max_i = i
		proxies.append(max_i)
	return proxies


#-------------------------------------------------------------------------
# ---- genetic algorithm ----


def cov_rate(x, U, rules):
	"""
    :param x: any instance
    :param U: a subset of the complete set of rules
    :return: a value between 0 and 1 that is coverage rate of x and U
	"""
	counter = 0
	for r in U:
		if x in rules[r]:
			counter += 1
	return counter / len(U)


def adj_cov_rate(x, U, rules):
	"""
    :param x: an instance
    :param U: a subset of the complete set of rules
    :param R: complete set of rules
    :return: Round(covRate(x, R) . |U|) / |U|
	"""
	return round(cov_rate(x, U, rules) * len(U)) / len(U)


def representativeness(R_eta, ids, rules):
	"""
    :param R_eta: set of cluster proxy rules R_eta over the clustering eta
    :param R: a set of induced rules
    :param U: a subset of complete set of rules
    :return: representativeness(R_eta, R)
	"""

	U = [i for i in range(len(rules))]
	counter = 0
    # todo: ids is complete set of instances
	for instance in ids:
		counter += abs(adj_cov_rate(instance, U, rules) - cov_rate(instance, R_eta, rules))
	return 1 - counter / len(ids)


def create_genome(clusters,n_clusters):
	return [random.choice(clusters[clus]) for clus in range(n_clusters)]


def mate(parent1, parent2, n_clusters,clusters):
	child = []
	for i in range(n_clusters):
		prob = random.random()
		if prob < 0.45:
			child.append(parent1[i])
		elif prob < 0.9:
			child.append(parent2[i])
		else:
			child.append(random.choice(clusters[i]))

	return child


def genetic_algo(clusters, n_clusters, ids, rules, silent):
	population_size = 20
	iteration_num = 100

	best_repr = 0
	best_genome = []

	pop_cut = (10 * population_size) // 100

	population = [create_genome(clusters,n_clusters) for i in range(population_size)]

	best_reps = []
	iters = [i for i in range(iteration_num)]
	for ite in range(iteration_num):
		population = sorted(population, key=lambda x: representativeness(x,ids,rules), reverse=True)


		best_rep = representativeness(population[0],ids,rules)
		if best_rep > best_repr:
			best_repr = best_rep
			best_genome = population[0]

		best_reps.append(best_rep)

        # directly go to next
		choose = (10 * population_size) // 100
		new_generation = population[0:choose]


		# print(new_generation)


		while len(new_generation) != population_size:
			parent1 = random.choice(population[:pop_cut])
			parent2 = random.choice(population[:pop_cut])
			# print(parent1,parent2)
			child = mate(parent1, parent2, n_clusters,clusters)
			new_generation.append(child)

		population = new_generation

	if not silent:
		plt.title("growth of representativeness via genetic algorithm")
		plt.xlabel("iteration number")
		plt.ylabel("representativeness of proxies")
		plt.plot(iters,best_reps)
		plt.show()


	return best_genome

#-------------------------------------------------------------------------
# ---- main functions ----

def find_proxies(silent=True):
	rules = read_id_set()
	ids = read_ids()
	info_rules = read_rules()


	clustering = find_best()
	labels = clustering.labels_
	clusters = create_clusters(labels)
	n_clusters = len(clusters)

	if not silent:
		print(clusters,n_clusters)
		# print(info_rules)

	proxies = initial_proxies(n_clusters,clusters,info_rules)
	best_rep_prox = representativeness(proxies,ids,rules)

	if not silent:
		print("proxies by quality", proxies, "repr:", best_rep_prox)
		print(info_rules.loc[proxies,:])



	best_genome = genetic_algo(clusters,n_clusters,ids,rules,silent)
	best_rep_gen = representativeness(best_genome,ids,rules)

	if not silent:
		print("proxies by genetic algorithm:", best_genome, "repr:", best_rep_gen)
		print(info_rules.loc[best_genome,:])


	best_proxies = proxies
	if best_rep_gen > best_rep_prox:
		best_proxies = best_genome

	print("final best proxies:", best_proxies)

	info_rules.loc[best_proxies,:].to_pickle("./chosen_rules_pysub")

	return best_proxies




if __name__=="__main__":
	find_proxies(False)

	# x = pd.read_pickle("./chosen_rules_pysub")

	# print(x)