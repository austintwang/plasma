import numpy as np 
import scipy.linalg.lapack as lp
import itertools

from .evaluator import Evaluator

class Finemap(object):

    NUM_CAUSAL_PRIOR_DEFAULT = 1.
    CROSS_CORR_PRIOR_DEFAULT = 0.
    IMBALANCE_HERIT_PRIOR_DEFAULT = 0.4
    TOTAL_EXP_HERIT_PRIOR_DEFAULT = 0.05
    LD_ADJ_PRIOR_DEFAULT = 1.
    PHASE_ERR_DEFAULT = 0.

    def __init__(self, **kwargs):
        self.num_snps = kwargs.get("num_snps", None)
        self.num_ppl = kwargs.get("num_ppl", None)
        self.as_only = kwargs.get("as_only", False)
        self.qtl_only = kwargs.get("qtl_only", False)
        self.force_defaults = kwargs.get("force_defaults", False)

        if self.force_defaults:
            self.num_causal_prior = self.NUM_CAUSAL_PRIOR_DEFAULT   
            self.cross_corr_prior = self.CROSS_CORR_PRIOR_DEFAULT
            self.imbalance_herit_prior = self.IMBALANCE_HERIT_PRIOR_DEFAULT
            self.total_exp_herit_prior = self.TOTAL_EXP_HERIT_PRIOR_DEFAULT
            self.ld_adj_prior = self.LD_ADJ_PRIOR_DEFAULT
            self.phase_err = self.PHASE_ERR_DEFAULT
        else:
            self.num_causal_prior = kwargs.get("num_causal_prior", self.NUM_CAUSAL_PRIOR_DEFAULT)   
            self.cross_corr_prior = kwargs.get("cross_corr_prior", self.CROSS_CORR_PRIOR_DEFAULT)
            self.imbalance_herit_prior = kwargs.get("imbalance_herit_prior", self.IMBALANCE_HERIT_PRIOR_DEFAULT)
            self.total_exp_herit_prior = kwargs.get("total_exp_herit_prior", self.TOTAL_EXP_HERIT_PRIOR_DEFAULT)
            self.ld_adj_prior = kwargs.get("ld_adj_prior", self.LD_ADJ_PRIOR_DEFAULT)
            self.phase_err = kwargs.get("phase_err", self.PHASE_ERR_DEFAULT)

        self.causal_status_prior = kwargs.get("causal_status_prior", None)
        self.imbalance_var_prior = kwargs.get("imbalance_var_prior", None)
        self.total_exp_var_prior = kwargs.get("total_exp_var_prior", None)

        self.imbalance_stats = kwargs.get("imbalance_stats", None)
        self.imbalance_corr = kwargs.get("imbalance_corr", None)
        self.total_exp_stats = kwargs.get("total_exp_stats", None)
        self.total_exp_corr = kwargs.get("total_exp_corr", None)
        self.corr_stats = kwargs.get("corr_stats", None)
        self.corr_shared = kwargs.get("corr_shared", None)
        self.cross_corr = kwargs.get("cross_corr", None)

        self.imbalance = kwargs.get("imbalance", None)
        self.phases = kwargs.get("phases", None)
        self.total_exp = kwargs.get("total_exp", None)
        self.genotypes_comb = kwargs.get("genotypes_comb", None)

        self.overdispersion = kwargs.get("overdispersion", None)
        self.imbalance_errors = kwargs.get("imbalance_errors", None)
        self.exp_errors = kwargs.get("exp_errors", None)

        self.counts_A = kwargs.get("counts_A", None)
        self.counts_B = kwargs.get("counts_B", None)

        self.hap_A = kwargs.get("hap_A", None)
        self.hap_B = kwargs.get("hap_B", None)
        self.hap_vars = kwargs.get("hap_vars", None)

        self.phi = kwargs.get("phi", None)
        self.beta = kwargs.get("beta", None)

        self._mean = None
        self._beta_normalizer = None

        self._covdiag_phi = None
        self._covdiag_beta = None

        self._haps_pooled = None

        self.evaluator = None

        # print(self.counts_A) ####
        # print(self.counts_B) ####
    
    def _calc_num_snps(self):
        """
        Infer number of markers from read counts vector 
        If already specified, defer to specified value
        """
        if self.num_snps is not None:
            return

        self.num_snps = np.shape(self.hap_A)[1]

    def _calc_num_ppl(self):
        """
        Infer sample size from haplotype matrix
        If already specified, defer to specified value
        """
        if self.num_ppl is not None:
            return

        self.num_ppl = np.size(self.counts_A)

    def _calc_causal_status_prior(self):
        """
        Infer causal status prior (gamma) as (# causal) / (# snps)
        If already specified, defer to specified value
        """
        if self.causal_status_prior is not None:
            return

        self._calc_num_snps()

        self.causal_status_prior = self.num_causal_prior / self.num_snps

    # def _calc_hap_vars(self):
    #   if self.hap_vars is not None:
    #       return

    #   haps_pooled = np.append(self.hap_A, self.hap_B, axis=0)
    #   self.hap_vars = np.var(haps_pooled, axis=0)

    def _calc_imbalance(self):
        """
        Calculate allelic imbalance phenotype (w) from read counts 
        If already specified, defer to specified value
        """
        if self.imbalance is not None:
            return

        # print(self.counts_A) ####
        # print(self.counts_B) ####
        imbalance_raw = np.log(self.counts_A) - np.log(self.counts_B)
        counts = self.counts_A + self.counts_B
        imbalance_adj = (
            imbalance_raw
            / (
                1
                + 1 / counts
                * (1 + self.overdispersion * (counts - 1))
            )
        )

        self.imbalance = (
            imbalance_adj
            + 1 / counts
            * np.sinh(imbalance_raw)
            * (1 + self.overdispersion * (counts - 1))
        )
    
    def _calc_phases(self):
        """
        Calculate genotype phasing (v) from haplotypes
        If already specified, defer to specified value
        """
        if self.phases is not None:
            return

        self.phases = self.hap_A - self.hap_B

    def _calc_total_exp(self):
        """
        Calculate total phenotype (y) from read counts
        If already specified, defer to specified value
        """
        if self.total_exp is not None:
            return

        self.total_exp = self.counts_A + self.counts_B

    def _calc_genotypes_comb(self):
        """
        Calculate genotype dosage (x) from read counts
        If already specified, defer to specified value
        """
        if self.genotypes_comb is not None:
            return

        self.genotypes_comb = self.hap_A + self.hap_B

    def _calc_corr_shared(self):
        """
        Calculate unified LD matrix (R) from genotypes
        If already specified, defer to specified value
        """
        if self.corr_shared is not None:
            return

        haps_pooled = np.append(self.hap_A, self.hap_B, axis=0)
        means = np.mean(haps_pooled, axis=0)
        haps_centered = haps_pooled - means
        cov = np.cov(haps_centered.T)
        covdiag = np.diag(cov)
        denominator = np.sqrt(np.outer(covdiag, covdiag))
        corr = cov / denominator
        self.corr_shared = np.nan_to_num(corr)
        np.fill_diagonal(self.corr_shared, 1.0)

    def _calc_imbalance_errors(self):
        """
        Calculate weights (predicted errors) for AI samples
        If already specified, defer to specified value
        """
        if self.imbalance_errors is not None:
            return

        self._calc_imbalance()

        imbalance_raw = np.log(self.counts_A) - np.log(self.counts_B)
        counts = self.counts_A + self.counts_B
        imbalance_adj = (
            imbalance_raw
            / (
                1
                + 1 / counts
                * (1 + self.overdispersion * (counts - 1))
            )
        )

        self.imbalance_errors = (
            2 / counts
            * (1 + np.cosh(imbalance_adj))
            * (1 + self.overdispersion * (counts - 1))
        )

    def _calc_phi(self):
        """
        Calculate AS effect sizes (phi)
        If already specified, defer to specified value
        """
        if self.phi is not None:
            return

        self._calc_imbalance_errors()
        self._calc_phases()
        self._calc_imbalance()

        phases = self.phases
        weights = 1 / self.imbalance_errors
        denominator = 1 / (phases.T * weights * phases.T).sum(1) 
        self.phi = denominator * np.matmul(phases.T, (weights * self.imbalance)) 

    def _calc_imbalance_stats(self):
        """
        Calculate AS z-scores (z_phi)
        If already specified, defer to specified value
        """
        if self.imbalance_stats is not None:
            return

        if self.qtl_only:
            self.imbalance_stats = np.empty(0)
            return

        self._calc_imbalance_errors()
        self._calc_phases()
        self._calc_imbalance()
        self._calc_phi()

        phases = self.phases
        weights = 1 / self.imbalance_errors
        denominator = 1 / (phases.T * weights * phases.T).sum(1) 
        phi = self.phi


        sqrt_weights = np.sqrt(weights)
        sum_weights = np.sum(weights)
        sum_weights_sq = np.sum(weights ** 2)
        sum_weights_sqrt = np.sum(sqrt_weights)
        residuals = (sqrt_weights * self.imbalance - sqrt_weights * (self.phases * phi).T).T
        remaining_errors = (
            np.sum(
                residuals * residuals - 1, 
                axis=0
            ) 
            / (sum_weights)
            + 4 * phi**2 * self.phase_err
        )

        varphi = (
            denominator 
            * denominator 
            * (
                (phases.T * weights**2 * phases.T).sum(1) 
                * remaining_errors 
                + (phases.T * weights * phases.T).sum(1)
            )
        )

        self.imbalance_stats = np.nan_to_num(phi / np.sqrt(varphi))

    def _calc_imbalance_corr(self):
        """
        Set LD for AS stats
        If already specified, defer to specified value
        """
        if self.imbalance_corr is not None:
            return

        if self.qtl_only:
            self.imbalance_corr = np.empty((0,0,))
        else:
            self._calc_corr_shared()
            self.imbalance_corr = self.corr_shared.copy()
        
    def _calc_beta(self):
        """
        Calculate QTL effect sizes (beta)
        If already specified, defer to specified value
        """
        if self.beta is not None:
            return

        self._calc_genotypes_comb()
        self._calc_total_exp()
        self._calc_num_snps()

        genotypes_comb = self.genotypes_comb
        genotype_means = np.mean(genotypes_comb, axis=0)
        exp_mean = np.mean(self.total_exp)
        genotypes_ctrd = genotypes_comb - genotype_means
        denominator = 1 / (genotypes_ctrd * genotypes_ctrd).sum(0)
        
        self.beta = denominator * genotypes_ctrd.T.dot(self.total_exp - exp_mean)
        self._mean = exp_mean
        self._beta_normalizer = denominator 

    def _calc_total_exp_errors(self):
        """
        Calculate QTL effect residuals
        If already specified, defer to specified value
        """
        if self.exp_errors is not None:
            return

        self._calc_beta()
        self._calc_num_ppl()

        residuals = (self.total_exp - self._mean - (self.genotypes_comb * self.beta).T).T
        self.exp_errors = np.sum(
            residuals * residuals, 
            axis=0
        ) / (self.num_ppl - 1)

    def _calc_total_exp_stats(self):
        """
        Calculate QTL association statistics (z_beta)
        If already specified, defer to specified value
        """
        if self.total_exp_stats is not None:
            return

        if self.as_only:
            self.total_exp_stats = np.empty(0)
            return

        self._calc_beta()
        self._calc_total_exp_errors()
        self._calc_num_snps()

        genotypes_comb = self.genotypes_comb
        genotype_means = np.mean(genotypes_comb, axis=0)
        exp_mean = np.sum(self.total_exp) / self.num_snps
        genotypes_ctrd = genotypes_comb - genotype_means

        genotypes_combT = genotypes_ctrd.T
        denominator = self._beta_normalizer

        # varbeta = denominator * denominator * (
        #   (genotypes_combT * genotypes_combT).sum(1) * self.exp_errors
        # )
        varbeta = denominator * self.exp_errors
        self.total_exp_stats = self.beta / np.sqrt(varbeta)

    def _calc_total_exp_corr(self):
        """
        Set LD for QTL stats
        If already specified, defer to specified value
        """
        if self.total_exp_corr is not None:
            return

        if self.as_only:
            self.total_exp_corr = np.empty((0,0,))
        else:
            self._calc_corr_shared()
            self.total_exp_corr = self.corr_shared.copy()

    def _calc_imbalance_var_prior(self):
        """
        Infer AS causal effect size hyperparameter
        If already specified, defer to specified value
        """
        if self.imbalance_var_prior is not None:
            return

        self._calc_num_ppl()

        coverage = np.mean(self.counts_A + self.counts_B)
        overdispersion = np.mean(self.overdispersion)
        imbalance = np.log(self.counts_A) - np.log(self.counts_B)
        ase_inherent_var = np.var(imbalance)

        ase_count_var = (
            2 / coverage
            * (
                1 
                + (
                    1
                    / (
                        1 / (np.exp(ase_inherent_var / 2))
                        + 1 / (np.exp(ase_inherent_var / 2)**3)
                        * (
                            (np.exp(ase_inherent_var * 2) + 1) / 2
                            - np.exp(ase_inherent_var)
                        )
                    )
                )
            )
            * (1 + overdispersion * (coverage - 1))
        )
        correction = ase_inherent_var / (ase_inherent_var + ase_count_var)
        self._imb_herit_adj = self.imbalance_herit_prior * correction

        # print(self.num_ppl) ####
        # print(self.num_causal_prior) ####
        # print(self._imb_herit_adj) ####
        self.imbalance_var_prior = (
            self.num_ppl 
            / self.num_causal_prior 
            * self._imb_herit_adj
            / (1 - self._imb_herit_adj)
        )

    def _calc_total_exp_var_prior(self):
        """
        Infer QTL causal effect size hyperparameter
        If already specified, defer to specified value
        """
        if self.total_exp_var_prior is not None:
            return

        self._calc_num_ppl()

        self.total_exp_var_prior = (
            self.num_ppl 
            / self.num_causal_prior 
            * self.total_exp_herit_prior 
            / (1 - self.total_exp_herit_prior)
        )

    def _calc_corr_stats(self):
        """
        Infer correlation between AS and QTL statistics
        If already specified, defer to specified value
        """
        if self.corr_stats is not None:
            return

        self._calc_imbalance_var_prior()

        self.corr_stats = self.cross_corr_prior * np.sqrt(
            (self.num_ppl / self.num_causal_prior)**2 
            * self.total_exp_herit_prior 
            * self._imb_herit_adj
            / (
                (
                    self.num_causal_prior * self.ld_adj_prior 
                    + self.total_exp_herit_prior * (self.num_ppl / self.num_causal_prior - 1)
                )
                * (
                    self.num_causal_prior * self.ld_adj_prior 
                    + self._imb_herit_adj * (self.num_ppl / self.num_causal_prior - 1)
                )
            )
        )
        # print(self.corr_stats) ####

    def _calc_cross_corr(self):
        """
        Infer causal effect correlation hyperparameter
        If already specified, defer to specified value
        """
        if self.cross_corr is not None:
            return
         
        self._calc_num_snps()

        if not self.qtl_only:
            self._calc_imbalance_stats()
            self._calc_imbalance_corr()
            self._calc_imbalance_var_prior()

        if not self.as_only:
            self._calc_total_exp_stats()
            self._calc_total_exp_corr()
            self._calc_total_exp_var_prior()

        if not (self.qtl_only or self.as_only):
            self._calc_corr_stats()

        if self.qtl_only:
            self.cross_corr = np.empty(shape=(self.num_snps,0))

        elif self.as_only:
            self.cross_corr = np.empty(shape=(0,self.num_snps))

        else:
            self.cross_corr = self.corr_shared * self.corr_stats

    def initialize(self, evaluator_cls=Evaluator):
        """
        Recursively calculate all required parameters and data from initial settings
        Create Evaluator instance for fine-mapping
        """
        self._calc_causal_status_prior()
        self._calc_imbalance_stats()
        self._calc_total_exp_stats()
        self._calc_imbalance_corr()
        self._calc_total_exp_corr()
        self._calc_cross_corr()

        self.evaluator = evaluator_cls(self)

    def search_exhaustive(self, min_causal, max_causal):
        """
        Conduct an exhaustive search over all possible causal configurations
        between min_causal and max_causal causal variants
        """
        m = self.num_snps
        configuration = np.zeros(m)
        for k in range(min_causal, max_causal + 1):
            for c in itertools.combinations(range(m), k):
                np.put(configuration, c, 1)
                self.evaluator.eval(configuration)
                configuration[:] = 0

    def search_shotgun(self, min_causal, max_causal, prob_threshold, streak_threshold, num_iterations):
        """
        Conduct an stochastic shotgun search over all possible causal configurations
        between min_causal and max_causal causal variants
        """
        m = self.num_snps
        configs = [np.zeros(m)]

        cumu_lposts = None
        streak = 0
        for i in range(num_iterations):
            lposts = []
            before_cumu_lposts = self.evaluator.cumu_lposts
            for c in configs:
                record_prob = np.count_nonzero(c) >= min_causal
                sel_lpost = self.evaluator.eval(c, save_result=record_prob)
                lposts.append(sel_lpost)

            after_cumu_lposts = self.evaluator.cumu_lposts
            if not after_cumu_lposts:
                diff_cumu_posts = 0
            elif before_cumu_lposts:
                diff_cumu_posts = np.exp(after_cumu_lposts) - np.exp(before_cumu_lposts)
            else:
                diff_cumu_posts = np.exp(after_cumu_lposts)

            if diff_cumu_posts <= prob_threshold:
                streak += 1
            else:
                streak = 0

            if streak >= streak_threshold:
                break

            lposts = np.array(lposts)
            lpostmax = np.max(lposts)
            posts = np.exp(lposts - lpostmax)
            dist = posts / np.sum(posts)
            selection = np.random.choice(np.arange(len(configs)), p=dist)
            configuration = configs[selection]

            num_causal = np.count_nonzero(configuration)
            configs = []
            for ind in range(m):
                val = configuration[ind]
                # Add causal variant
                if (val == 0) and (num_causal < max_causal):
                    neighbor = configuration.copy()
                    neighbor[ind] = 1
                    configs.append(neighbor)
                # Remove causal variant
                elif val == 1:
                    neighbor = configuration.copy()
                    neighbor[ind] = 0
                    configs.append(neighbor)
                # Swap status with other variants
                for ind2 in range(ind+1, m):
                    val2 = configuration[ind2]
                    if val2 != val:
                        neighbor = configuration.copy()
                        neighbor[ind] = val2
                        neighbor[ind2] = val
                        configs.append(neighbor)

    def get_probs(self):
        return self.evaluator.get_probs()

    def get_probs_sorted(self):
        probs = list(self.get_probs().items())
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs

    def get_causal_set(self, confidence, heuristic="max_increase"):
        results_exp = self.get_probs()
        
        if heuristic == "max_ppa":
            causal_set = np.ones(self.num_snps)
            conf_sum = 1.

            snp_sets = {i: set() for i in range(self.num_snps)}
            for c in results_exp.keys():
                for i in range(self.num_snps):
                    if c[i] == 1:
                        snp_sets[i].add(c)

            # print([(k, len(v)) for k, v in snp_sets.items()]) ####

            ppas = self.get_ppas()
            ppas_sort = np.argsort(ppas)
            for i in ppas_sort:
                conf_sum_after = conf_sum - sum([results_exp[s] for s in snp_sets[i]])
                # print([results_exp[s] for s in snp_sets[i]]) ####
                # print(sum([results_exp[s] for s in snp_sets[i]])) ####
                if conf_sum_after > confidence:
                    remove_set = snp_sets.pop(i)
                    for s in snp_sets.values():
                        s -= remove_set

                    causal_set[i] = 0
                    conf_sum = conf_sum_after
                else:
                    break

        elif heuristic == "max_increase":
            causal_set = np.zeros(self.num_snps)
            conf_sum = results_exp.get(tuple(causal_set), 0.)
            distances = {}
            causal_extras = {}
            for k in results_exp.keys():
                causals = set(ind for ind, val in enumerate(k) if val == 1)
                distances.setdefault(sum(k), set()).add(k)
                causal_extras[k] = causals

            while conf_sum < confidence:
                dist_ones = distances[1]
                neighbors = {}
                for i in dist_ones:
                    diff_snp = next(iter(causal_extras[i]))
                    neighbors.setdefault(diff_snp, 0)
                    neighbors[diff_snp] += results_exp[i]

                max_snp = max(neighbors, key=neighbors.get)
                causal_set[max_snp] = 1
                conf_sum += neighbors[max_snp]
                # print(conf_sum) ####

                diffs = {}
                for k, v in distances.items():
                    diffs[k] = set() 
                    for i in v:
                        if i[max_snp] == 1:
                            diffs[k].add(i)
                            if k == 1:
                                causal_extras.pop(i)
                            else:
                                causal_extras[i].remove(max_snp)

                for k, v in diffs.items():
                    distances[k] -= v
                    if k > 1:
                        distances.setdefault(k-1, set())
                        distances[k-1] |= v

        # print(causal_set) ####
        return list(causal_set) 

    def get_ppas(self):
        ppas = []
        for i in range(self.num_snps):
            ppa = 0
            for k, v in self.get_probs().items():
                if k[i] == 1:
                    ppa += v
            ppas.append(ppa)
        return np.array(ppas)

    def get_size_probs(self):
        size_probs = np.zeros(self.num_snps)
        for k, v in self.get_probs().items():
            num_snps = np.count_nonzero(k)
            size_probs[num_snps] += v
        return size_probs

    def reset_mapping(self):
        self.evaluator.reset()

    def coloc_clpps(self, other):
        return self.get_ppas() * other.get_ppas()

    def coloc_hyps(self, other):
        ppas1 = self.get_ppas()
        ppas2 = other.get_ppas()

        h4 = np.sum(ppas1 * ppas2)
        h3 = np.sum(ppas1) * np.sum(ppas2) - h4
        h0 = (1 - np.sum(ppas1)) * (1 - np.sum(ppas2))
        h1 = np.sum(ppas1) * (1 - np.sum(ppas2))
        h2 = (1 - np.sum(ppas1)) * np.sum(ppas2)
        
        return h0, h1, h2, h3, h4

    @classmethod
    def multi_coloc_clpps(cls, instances):
        clpps = 1.
        for i in instances:
            clpps *= i.get_ppas()
