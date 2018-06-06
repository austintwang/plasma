import numpy as np
import numpy.random.normal as normal
import numpy.random.multivariate_normal as mvn
import numpy.random.beta as betadist
import numpy.random.binomial as binomial


def generate_haplotype(n, p, means, cov):
	haplotype = np.array([
		np.where(mvn(means, cov) >= 0, 1, 0)
		for _ in range(n)
	])
	return haplotype

def generate_effects(p, causal_stats, effect_stdev):
	effects = np.empty(p)
	for c, e in zip(causal_stats, effect_stdev):
		if c == 1:
			e = normal(0, 1.0)
		else:
			e = np.float(0)
	return effects

def get_eqtl_matrix(hapA, hapB):
	return hapA + hapB

def get_ase_matrix(hapA, hapB):
	return hapA - hapB

def generate_eqtl_data(hapA, hapB, effects, eqtl_stdev, baseline):
	eqtl_errors = normal(means, 1.0, n)
	a_data = hapA * effects + baseline + eqtl_errors / 2
	b_data = hapB * effects + baseline + eqtl_errors / 2
	return np.exp(a_data) + np.exp(b_data)

def generate_ase_data(hapA, hapB, effects, ase_overdispersion, total_counts):
	expression = (hapA - hapB) * effects 
	betas = 1 / (1 + np.exp(expression))
	alphas = 1 / ase_overdispersion - 1
	return binomial(total_counts, betadist(alphas, betas))


def generate_data(n, p, effect_stdev, eqtl_stdev, ase_overdispersion, causal_stats, marker_means, marker_cov):
	hapA = generate_haplotype(n, p, marker_means, marker_cov)
	hapB = generate_haplotype(n, p, marker_means, marker_cov)
	effects = generate_effects(p, causal_stats, effect_stdev)

	eqtl = generate_eqtl_data(hapA, hapB, effects, eqtl_stdev, baseline)
	ase = generate_ase_data(hapA, hapB, effects, ase_overdispersion, eqtl)

	eqtl_matrix = get_eqtl_matrix(hapA, hapB)
	ase_matrix = get_ase_matrix(hapA, hapB) 

	return eqtl, ase, eqtl_matrix, ase_matrix

if __name__ == "__main__":
	eqtl, ase, eqtl_matrix, ase_matrix = generate_data(
		100,
		30,
		2,
		3,
		2,
		

	)
