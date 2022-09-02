import datetime
import json
import os
import subprocess
import warnings

import more_itertools as mit
import num2words as n2w
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as skm
import tabulate
from matplotlib import pyplot as plt
from statsmodels.tsa.api import VAR

from CausalA import Tools as TB, TS_Sim

sns.set(rc={"figure.dpi": 125, 'savefig.dpi': 300})


class Granger:
    """
    Class that performs the different Granger tests with regard to the type (bi-variate or multivariate) given.
    Includes transformation of p-values and the export as well as import procedure to and from R.
    """
    _allowed_tt = ["bi-variate", "multivariate", "all_on_one"]
    _file_names = {0: "time_series.csv",
                   1: "transfer_data.json",
                   2: "R_data.json"}
    _packages = {"bi-variate": ["statsmodels", "vars", "lmtest", "VLTimeCausality", "NlinTS"],
                 "multivariate": ["statsmodels", "vars", "bruceR"],
                 "all_on_one": ["statsmodels", "vars"]}
    _marker = ["o", "*", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h", "v", "^", "<", ">", "1",
               "2", "3", "4", "|", "_", ".", ",", ]
    granger_pvalues = None
    granger_results = None
    granger_confusion_matrices = None

    def __init__(self, test_type, ts_data, significance_level, cwd=None, folder_name=None) -> None:
        self.__is_valid(test_type, ts_data)
        self.sig_lvl = significance_level
        self.folder_name = folder_name
        self.cwd = cwd if cwd is not None else self.ts.cwd
        self.__init_attr()
        self.__init_statsm()
        self.__ex_im_r()

    def __is_valid(self, test_type, ts_data):
        if test_type not in self._allowed_tt:
            raise Warning(f"Test type {test_type} is not supported. Supported test types are: {self._allowed_tt}")
        self.test_type = test_type

        if isinstance(ts_data, TS_Sim.TS_Sim):
            self.ts = ts_data
        else:
            raise Warning(
                    f"Please provide the time series data as TS_Sim.TS_Sim and not {type(ts_data)}")

    def __gen_dir(self):
        if self.folder_name is None:
            self.fpath = f"{self.ts.cwdf}/single/{self.test_type}"
        else:
            self.fpath = f"{self.ts.cwdf}/single/{self.test_type}/{self.folder_name}"
        if not os.path.exists(self.fpath):
            os.makedirs(self.fpath)

    def __init_attr(self):
        self.__n_comb()
        self.__ts_combination()
        self.__init_res()
        self.__gen_dir()

    def __init_res(self):
        tmp_dict = dict(zip(self._packages[self.test_type], np.repeat(np.nan, repeats=len(self._packages[
                                                                                              self.test_type]))))
        self.granger_pvalues = tmp_dict.copy()
        self.granger_results = tmp_dict.copy()
        self.granger_confusion_matrices = tmp_dict.copy()
        if self.test_type == "multivariate":
            self.multivariate_acc_approx = tmp_dict.copy()

    def __n_comb(self):
        self._num_comb = {
            "bi-variate": self.ts.k - 1,
            "multivariate": 2 ** (self.ts.k - 1) - 1,
            "all_on_one": 1
        }

    def __ts_combination(self):
        """
        Calculates the number of test combinations per type.
        """
        if self.test_type == "bi-variate":
            self.ts_combinations = pd.DataFrame(
                    [self.ts.ts_names[np.delete(np.arange(self.ts.k), i)] for i in np.arange(
                            self.ts.k)]).T
        elif self.test_type == "multivariate":
            self.ts_combinations = pd.DataFrame(
                    [[[*j] for j in mit.powerset(mit.unique_everseen(np.delete(self.ts.ts_names, t)))][1:] for t in
                     np.arange(self.ts.k)]).T
        elif self.test_type == "all_on_one":
            self.ts_combinations = pd.DataFrame([np.delete(self.ts.ts_names, i)] for i in np.arange(self.ts.k)).T

    def __init_statsm(self):
        self.__VAR()
        self.__statsm_run()
        self.__statsm_gc_res()
        if self.test_type == "bi-variate":
            self.__run_partial_gc()

    def __VAR(self):
        self.VAR = VAR(self.ts.get_ts())

    def __statsm_run(self):
        self.statsm_gc_objects = np.empty((self.ts.p, self._num_comb[self.test_type], self.ts.k), dtype=object)
        for l in np.arange(self.ts.p):
            tmp_var = self.VAR.fit(l + 1)
            # read from left to right to get the lag-coefficients for a specific time series
            for t in np.arange(self.ts.k):
                for r, i in enumerate(self.ts_combinations[t]):
                    self.statsm_gc_objects[l][r, t] = tmp_var.test_causality(caused=self.ts.ts_names[t], causing=i,
                                                                             signif=self.sig_lvl)

    def __run_partial_gc(self):
        self.gc_partial_pvalues = np.ones((self.ts.p, self._num_comb[self.test_type], self.ts.k))
        for i, d in enumerate(self.gc_partial_pvalues):
            for ts in np.arange(self.ts.k):
                d[:, ts] = [
                    TB.Tools.partial_granger_causality(x=self.ts.df_ts.iloc[:, j], y=self.ts.df_ts.iloc[:, ts],
                                                       p=i + 1) for j in np.delete(np.arange(self.ts.k), ts)]
        self.gc_partial_infl = np.less(self.gc_partial_pvalues, self.sig_lvl).astype(int)

    def __statsm_gc_res(self):
        self.granger_pvalues["statsmodels"] = TB.Tools.deep_inplace_replace(self.statsm_gc_objects, "pvalue")
        self.granger_results["statsmodels"] = self.__gc_transform_pvalues(self.granger_pvalues["statsmodels"])

    def __gc_transform_pvalues(self, p_values):
        # encodes the respective pvalue to 1 if it is significant and to 0 if it is not
        gc_significant = np.less(p_values, self.sig_lvl).astype(int)
        gc_res = [np.zeros((i + 1, self.ts.k, self.ts.k), dtype=int) for i in np.arange(gc_significant.shape[0])]
        s_sel = np.array([np.delete(np.arange(self.ts.k), i) for i in np.arange(self.ts.k)]).flatten().reshape(
                self.ts.k, -1)
        if self.test_type == "bi-variate":
            for l in np.arange(gc_significant.shape[0]):
                # TODO check
                gc_res[l][:, np.repeat(np.arange(self.ts.k), self.ts.k - 1).reshape(self.ts.k, -1), s_sel] = \
                    gc_significant[l].T
        elif self.test_type == "multivariate":
            warnings.warn(
                    "Multivariate comparison is not feasible due to a lack of true values and transitivity properties. "
                    "One can only compare the resulting p-values between libraries.")
        elif self.test_type == "all_on_one":
            for l in np.arange(gc_significant.shape[0]):
                tmp_bool = np.equal(gc_significant[l], 1).flatten()
                tmp_r_sel = s_sel[tmp_bool, :]
                tmp_c_sel = np.array([np.repeat(i, repeats=self.ts.k - 1).flatten() for i in np.arange(self.ts.k)])[
                            tmp_bool, :]
                if tmp_r_sel.shape != tmp_c_sel.shape:
                    warnings.warn(
                            f"Shape differs between tmp_r_sel: {tmp_r_sel.shape} and tmp_c_sel: {tmp_c_sel.shape}")
                gc_res[l][:, tmp_c_sel, tmp_r_sel] = 1
        return gc_res

    def __export_data(self):
        # dictionary with all relevant parameters
        transfer_data = {"T": self.ts.T, "k": self.ts.k, "p": self.ts.p, "ts_names": self.ts.ts_names.tolist(),
                         "cwd": self.cwd, "ts_combinations": self.ts_combinations.T.values.tolist() if
            self.test_type != "all_on_one" else [list(i) for i in self.ts_combinations.to_numpy().flatten()],
                         "r_filename": self._file_names[2], "test_type": self.test_type,
                         "c_len": self._num_comb[self.test_type],
                         "packages": TB.Tools.rm_el(self._packages[self.test_type], "statsmodels"),
                         "seed": self.ts.seed}
        # exports all parameters as .json
        with open(f"{self.cwd}/{self._file_names[1]}", "w") as file:
            json.dump(transfer_data, file, sort_keys=True, indent=4)
        # exports the time series data as .csv
        self.ts.get_ts().to_csv(f"{self.cwd}/{self._file_names[0]}", index=False)

    def __ex_im_r(self):
        self.__export_data()
        self.__call_r()
        self.__open_r_data()
        self.__process_r_data()
        if self.test_type == "multivariate":
            self.__comp_multivariate_res()
        self.__gen_conf_mat()

    def __call_r(self):
        subprocess.call(f"Rscript --vanilla main.R {self._file_names[0]} {self._file_names[1]}", shell=True,
                        cwd=self.cwd)

    def __open_r_data(self):
        file = open(f"{self.cwd}/{self._file_names[2]}")
        self.r_data = json.load(file)

    def __process_r_data(self):
        for n, d in self.r_data.items():
            tmp_pvalues = np.asarray(d).reshape((self.ts.p, -1, self.ts.k), order="F")
            self.granger_pvalues[n] = tmp_pvalues
            self.granger_results[n] = self.__gc_transform_pvalues(tmp_pvalues)

    def __gen_conf_mat(self):
        """
        Calculates the confusion matrices for each package.
        """
        for n, d in self.granger_results.items():
            tmp_pred = list([TB.Tools.rm_diag(e) for e in d])
            tmp_act = TB.Tools.rm_diag(self.ts.D)
            self.granger_confusion_matrices[n] = TB.Tools.confussion_matrices(
                    predictions=tmp_pred,
                    actual=tmp_act)

    def __comp_multivariate_res(self):
        """
        Approximates the accuracy rate per time series.
        """
        for n, d in self.granger_pvalues.items():
            self.multivariate_acc_approx[n] = [np.less(d[l], self.sig_lvl).astype(int).mean(axis=0) for l in np.arange(
                    self.ts.p)]
        self.approx_actual = [TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.D))[:l + 1].mean(axis=0).mean(axis=0)
                              for l in np.arange(self.ts.p)]

    def plot_multi_comp(self, add=False, cwd=None, verbose=True):
        with sns.axes_style("darkgrid"):
            fig, axes = plt.subplots(nrows=self.ts.k, figsize=(8, 5 * self.ts.k))
            if not add:
                fig.suptitle(f"Detected causal influence on the specific time series per package\n", y=.99)
            else:
                fig.suptitle(f"Detected causal influence on the specific time series per package ({add})\n", y=.99)
            x_ticks = np.arange(1, self.ts.p + 1)
            id = dict(zip(self.multivariate_acc_approx.keys(),
                          np.arange(len(self.multivariate_acc_approx.keys()))))
            for i, ax in enumerate(axes.flat):
                for n, d in self.multivariate_acc_approx.items():
                    ax.plot(x_ticks, [d[t][i] for t in np.arange(self.ts.p)], marker=self._marker[id[n] + 1],
                            linestyle="--",
                            alpha=.5, label=f"{n}")
                ax.plot(x_ticks, [t[i] for t in self.approx_actual], label=f"Approximation of actual causal "
                                                                           f"influence on t{i + 1}", marker="o",
                        alpha=.75)
                ax.legend(loc="best")
                ax.set_ylim(-0.05, 1.05)
                ax.set_title(f"{n2w.num2words(i + 1, to='ordinal_num')} time series")
            plt.xlabel("Lags")
            plt.xticks(x_ticks)
            plt.tight_layout()
            _path = self.fpath if cwd is None else cwd
            plt.savefig(fname=f"{_path}/multivariate_comp_{str(datetime.datetime.now())}.pdf", format="pdf")
            if verbose:
                fig.show()
            else:
                plt.close(fig)

    def __calc_partial_acc(self):
        self.part_acc_lag = np.zeros(self.ts.p)
        for l in np.arange(self.ts.p):
            tmp_cm_part = skm.confusion_matrix(y_pred=self.gc_partial_infl[l].flatten(),
                                               y_true=TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.D))[l].flatten(),
                                               labels=[0, 1])
            self.part_acc_lag[l] = np.divide(np.trace(tmp_cm_part), tmp_cm_part.sum())

    def plot_partial_comp(self, add=False, cwd=None, verbose=True):
        self.__calc_partial_acc()
        not_part_lag = np.zeros(self.ts.p)
        n = len(self.granger_confusion_matrices)
        _x_ticks = np.arange(1, self.ts.p + 1)
        fig, axes = plt.subplots(nrows=n, figsize=(self.ts.p * 1.5, 5 * n), tight_layout=True)
        if not add:
            plt.suptitle(
                    f"Accuracy of normal and partial bi-variate Granger-causality results for each lag", y=.995)
        else:
            plt.suptitle(
                    f"Accuracy of normal and partial bi-variate Granger-causality results for each lag {add})", y=.995)
        ax = dict(zip(self.granger_confusion_matrices.keys(), axes.flat))
        with sns.axes_style("darkgrid"):
            for k, v in self.granger_confusion_matrices.items():
                for l in np.arange(self.ts.p):
                    not_part_lag[l] = np.divide(np.trace(self.granger_confusion_matrices[k][l]),
                                                self.granger_confusion_matrices[k][l].sum())
                ax[k].plot(_x_ticks, self.part_acc_lag, marker="o", label="Accuracy of partial bi-variate GC",
                           alpha=0.5, ls="solid", lw=2)
                ax[k].plot(_x_ticks, not_part_lag, marker="d", label=f"Accuracy of normal bi-variate GC ({k})",
                           alpha=0.5, ls="solid", lw=2)
                ax[k].plot(_x_ticks, self.ts.a, marker="+", label="Strength decrease of the coefficients", alpha=0.5,
                           ls="dotted")
                ax[k].set_xlabel("Lags")
                ax[k].set_ylabel("Prediction accuracy")
                ax[k].legend(loc="best")
                ax[k].set_ylim(0, 1.05)
        _path = self.fpath if cwd is None else cwd
        TB.Tools.create_dir_if_missing(dir=f"{_path}/accuracy")
        plt.savefig(fname=f"{_path}/accuracy/partial_comp_{str(datetime.datetime.now())}.pdf", format="pdf")
        if verbose:
            fig.show()
        else:
            plt.close(fig)

    def plot_accuracy(self, verbose=True):
        for n, cm in self.granger_confusion_matrices.items():
            TB.Tools.plot_true_detections(confusion_matrices=cm, test_type=self.test_type, package=n,
                                          decay=self.ts.a,
                                          cwd=TB.Tools.create_dir_if_missing(dir=f"{self.fpath}/accuracy",
                                                                             return_dir=True),
                                          verbose=verbose)

    def plot_correlations(self, verbose=True, sig_lvl=0.05, print_corr_summary=True):
        col = ["Package", "Lags", "Pearson", "Pearson p-value", "Point biserial", "Point biserial p-value"]
        self.corr_panel = pd.DataFrame(np.zeros((0, len(col))), columns=col)
        for n, d in self.granger_pvalues.items():
            pear, point = TB.Tools.corr_test(true_inf=TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.D)), gc_pvalues=d,
                                             true_coef=TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.A)),
                                             sig_lvl=sig_lvl,
                                             package=n, decay=self.ts.a,
                                             cwd=TB.Tools.create_dir_if_missing(dir=f"{self.fpath}/correlation",
                                                                                return_dir=True),
                                             verbose=verbose)
            tmp_df = pd.DataFrame([[n, l + 1, pear[l][0][0], pear[l][0][1], point[l][0][0], point[l][0][1]] for l in
                                   np.arange(self.ts.p)], columns=col)
            self.corr_panel = self.corr_panel.append(tmp_df)
        if print_corr_summary:
            self.__print_corr_summary(verbose=verbose)

    def __print_corr_summary(self, corr_type="Pearson", verbose=True):
        self.corr_panel = self.corr_panel.fillna(0)
        std = self.corr_panel.groupby("Package")[corr_type].std()
        mean = self.corr_panel.groupby("Package")[corr_type].mean()
        min = self.corr_panel.groupby("Package")[corr_type].min()
        max = self.corr_panel.groupby("Package")[corr_type].max()

        std_all = self.corr_panel.iloc[:, self.corr_panel.columns == corr_type].std()
        mean_all = self.corr_panel.iloc[:, self.corr_panel.columns == corr_type].mean()
        min_all = self.corr_panel.iloc[:, self.corr_panel.columns == corr_type].min()
        max_all = self.corr_panel.iloc[:, self.corr_panel.columns == corr_type].max()

        cols = ["Mean", "Std.", "Min", "Max"]
        all = pd.DataFrame(pd.concat([mean_all, std_all, min_all, max_all], axis=1, keys=cols), columns=cols)
        all.index = ["All packages"]
        tmp_ = pd.concat([mean, std, min, max], axis=1, keys=cols)
        df = tmp_.append(all)
        fig, ax = plt.subplots()
        plt.axhline(y=all.iloc[0, 0], label="Overall mean", ls="dashed", alpha=0.5)
        sns.boxenplot(x="Package", y=corr_type, data=self.corr_panel, ax=ax)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.fpath}/correlation/{corr_type}_corr_boxplot.pdf", format="pdf")
        if verbose:
            fig.show()
        else:
            plt.close(fig)

        tab = tabulate.tabulate(df, headers="keys", tablefmt="latex")
        with open(f"{self.fpath}/correlation/{corr_type}_corr_summary.txt", "a") as f:
            print(tab, file=f)

    def plot_part_corr(self, verbose=True, sig_lvl=0.05):
        TB.Tools.corr_test(true_inf=TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.D)), gc_pvalues=self.gc_partial_pvalues,
                           true_coef=TB.Tools.t_inside(TB.Tools.rm_diag(self.ts.A)), sig_lvl=sig_lvl,
                           package="Own implementation", decay=self.ts.a,
                           cwd=TB.Tools.create_dir_if_missing(dir=f"{self.fpath}/correlation", return_dir=True),
                           verbose=verbose, partial=True)

    def get_key_attr(self):
        return self.granger_pvalues, self.granger_confusion_matrices

    def plot_conf(self, verbose=False):
        for n, d in self.granger_confusion_matrices.items():
            TB.Tools.plot_cm(confusion_matrices=d,
                             cwd=TB.Tools.create_dir_if_missing(dir=f"{self.fpath}/confusion_matrices",
                                                                return_dir=True),
                             package_name=n, verbose=verbose)
