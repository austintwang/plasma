import numpy as np 
import scipy.linalg.lapack as lp

class Evaluator(object):
    
    def __init__(self, fm):
        if fm.qtl_only:
            self.num_snps_imbalance = 0
        else:
            self.num_snps_imbalance = fm.num_snps

        if fm.as_only:
            self.num_snps_total_exp = 0
        else:
            self.num_snps_total_exp = fm.num_snps

        self.num_snps_combined = self.num_snps_imbalance + self.num_snps_total_exp
        self.num_snps = max(self.num_snps_imbalance, self.num_snps_total_exp)

        self.causal_status_prior = fm.causal_status_prior

        self.imbalance_stats = fm.imbalance_stats
        self.total_exp_stats = fm.total_exp_stats

        self.imbalance_corr = fm.imbalance_corr
        self.total_exp_corr = fm.total_exp_corr
        self.cross_corr = fm.cross_corr

        self.imbalance_var_prior = fm.imbalance_var_prior
        self.total_exp_var_prior = fm.total_exp_var_prior
        self.cross_corr_prior = fm.cross_corr_prior
        # self.cross_cov_prior = (
        #     self.cross_corr_prior 
        #     * np.sqrt(fm.imbalance_var_prior * fm.total_exp_var_prior)
        # )

        self.imbalance_var_inv = (
            1 / (fm.imbalance_var_prior * (1 - fm.cross_corr_prior ** 2))
        )
        self.total_exp_var_inv = (
            1 / (fm.total_exp_var_prior * (1 - fm.cross_corr_prior ** 2))
        )
        self.cross_cov_inv = (
            -fm.cross_corr_prior 
            / (
                np.sqrt(fm.imbalance_var_prior * fm.total_exp_var_prior) 
                * (1 - fm.cross_corr_prior ** 2)
            )
        )

        self.prior_cov_inv = np.block([
            [
                np.eye(self.num_snps_imbalance) * self.imbalance_var_inv, 
                np.eye(self.num_snps_imbalance, self.num_snps_total_exp) * self.cross_cov_inv
            ],
            [
                np.eye(self.num_snps_total_exp, self.num_snps_imbalance) * self.cross_cov_inv, 
                np.eye(self.num_snps_total_exp) * self.total_exp_var_inv
            ]
        ])

        # print(self.imbalance_corr) ####
        # print(self.cross_corr.T) ####
        # print(self.cross_corr) ####
        # print(self.total_exp_corr) ####
        self.corr = np.block([
            [self.imbalance_corr, self.cross_corr.T],
            [self.cross_corr, self.total_exp_corr]
        ])

        self.inv_term = self.prior_cov_inv + self.corr

        self.stats = np.append(self.imbalance_stats, self.total_exp_stats)

        if self.num_snps_imbalance == 0:
            self.ldet_prior = np.log(self.total_exp_var_prior)
        elif self.num_snps_total_exp == 0:
            self.ldet_prior = np.log(self.imbalance_var_prior)
        else:
            self.ldet_prior = (
                np.log(self.imbalance_var_prior)
                + np.log(self.total_exp_var_prior)
                + np.log(1 - self.cross_corr_prior ** 2)
            )

        # print(self.imbalance_stats) ####
        # print(self.total_exp_stats) ####
        self.valid_entries = np.append(
            np.isfinite(self.imbalance_stats), 
            np.isfinite(self.total_exp_stats)
        )

        self.snp_map = np.append(
            np.arange(self.num_snps_imbalance),
            np.arange(self.num_snps_total_exp)
        )

        self.null_config = np.zeros(self.num_snps)

        # Create structure for storing unnormalized results for each causal configuration
        self.results = {}
        self.results_unsaved = {}
        self.cumu_lposts = None

    @staticmethod
    def _lbf(cov_term, stats, ldet_prior):
        ltri = np.linalg.cholesky(cov_term)
        ldet_term = np.log(np.prod(np.diagonal(ltri))) * 2
        temp = np.asfortranarray(stats[:, np.newaxis])
        lp.dtrtrs(ltri, temp, lower=1, overwrite_b=1)
        return (-ldet_term - ldet_prior * stats.size + np.sum(temp ** 2)) / 2

    def eval(self, configuration, lprior=None, save_result=True):
        key = tuple(configuration.tolist())

        if key in self.results:
            return self.results[key] 
        elif key in self.results_unsaved:
            if save_result:
                self.results[key] = self.results_unsaved[key] 
            return self.results_unsaved[key] 

        if lprior is None:
            num_causal = np.count_nonzero(configuration)
            lprior = (
                np.log(self.causal_status_prior) * num_causal
                + np.log(1 - 2 * self.causal_status_prior) * (self.num_snps - num_causal)
            )

        if np.array_equal(configuration, self.null_config):
            res = lprior

        else:
            configuration_bool = (configuration == 1)
            selection = np.logical_and(configuration_bool[self.snp_map], self.valid_entries)
            if not np.any(selection):
                res = lprior
                self.results[key] = res
                return res

            selection_2d = np.ix_(selection, selection)
            inv_term_subset = self.inv_term[selection_2d]
            stats_subset = self.stats[selection]

            lbf = self._lbf(inv_term_subset, stats_subset, self.ldet_prior)

            res = lbf + lprior

        if save_result:
            self.results[key] = res
            if self.cumu_lposts is None:
                self.cumu_lposts = res
            elif res - self.cumu_lposts < 100:
                self.cumu_lposts += np.log(1 + np.exp(res - self.cumu_lposts))
            else:
                self.cumu_lposts = res + np.log(1 + np.exp(self.cumu_lposts - res))
        else:
            self.results_unsaved[key] = res
        return res

    def get_probs(self):
        probs = {}
        total_lpost = self.cumu_lposts

        for k, v in self.results.items():
            lprob = v - total_lpost
            probs[k] = np.exp(lprob)
        return probs

    def reset(self):
        self.results = {}

