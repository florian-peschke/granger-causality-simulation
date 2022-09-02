import copy
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tabulate import tabulate

from CausalA import Granger, Tools as tb, TS_Sim

sns.set(rc={"figure.dpi": 125, 'savefig.dpi': 300})


class SensA:
    """
    Class to perform the sensitivity analyses.
    """
    _init_corr_col = ["Package", "Type", "Corr", "p", "Lag"]
    _init_acc_col = ["Package", "Accuracy", "Lag"]
    _marker = ["o", "*", "x", "X", "+", "P", "s", "D", "d", "p", "H", "h", "v", "^", "<", ">", "1",
               "2", "3", "4", "|", "_", ".", ",", ]

    def __init__(self, parameter_name, parameter_values, test_type, ts_data, significance_level,
                 cwd=None, ts_update=True, iterate_sig_lvl=False, folder_name=None) -> None:
        self.tmp_res = None
        self.parameter_name = parameter_name
        self.parameter_values = parameter_values
        self.ts_update = ts_update
        self.folder_name = folder_name
        self.iterate_sig_lvl = iterate_sig_lvl
        if isinstance(ts_data, TS_Sim.TS_Sim):
            self.ts = ts_data
        else:
            raise Warning(
                    f"Please provide the time series data as TS_Sim.TS_Sim and not {type(ts_data)}")
        self.test_type = test_type
        self.significance_level = significance_level
        self.cwd = cwd if cwd is not None else self.ts.cwd
        self.__gen_dir()
        self._corr_col = self._init_corr_col + [parameter_name]
        self._acc_col = self._init_acc_col + [parameter_name]
        self.res = list()
        self.ts_res = list()
        self.granger_objects = list()
        self.df_corr = pd.DataFrame(np.zeros((0, len(self._corr_col))), columns=self._corr_col)
        self.df_acc = pd.DataFrame(np.zeros((0, len(self._acc_col))), columns=self._acc_col)

    def __gen_dir(self):
        if self.folder_name is None:
            self.fpath = f"{self.ts.cwdf}/sensitivity/{self.test_type}"
        else:
            self.fpath = f"{self.ts.cwdf}/sensitivity/{self.test_type}/{self.folder_name}"
        tb.Tools.create_dir_if_missing(dir=self.fpath)

    def run_gc(self):
        for i, v in enumerate(self.parameter_values):
            tmp_ts = copy.deepcopy(self.ts)
            if self.ts_update:
                tmp_ts.update(self.parameter_name, v)
            if self.iterate_sig_lvl:
                self.significance_level = v
            self.ts_res.append(tmp_ts)
            tmp_gc = Granger.Granger(test_type=self.test_type,
                                     significance_level=self.significance_level,
                                     cwd=self.cwd, ts_data=tmp_ts)
            self.granger_objects.append(tmp_gc)
            self.res.append(tmp_gc.get_key_attr())
            self.__process(*tmp_gc.get_key_attr(), v=v, i=i)

    def __process(self, pval, conf, v, i):
        for k, val in pval.items():
            self.df_corr = self.df_corr.append(self.__calc_correlations(val, k, v, i))
        for k, val in conf.items():
            self.df_acc = self.df_acc.append(self.__calc_acc(val, k, v, i))

    def __calc_acc(self, conf, package, v, i):
        tmp_df = pd.DataFrame(np.zeros((self.ts_res[i].p, len(self._acc_col))), columns=self._acc_col)
        tmp_df.iloc[:, :] = [[package, np.divide(np.trace(c), np.sum(c)), i + 1, v] for i, c in
                             enumerate(conf)]
        return tmp_df

    def __calc_correlations(self, pvalues, package, v, i):
        tmp_df = pd.DataFrame(np.zeros((2 * self.ts_res[i].p, len(self._corr_col))), columns=self._corr_col)
        pb, pears = tb.Tools.calc_corr(true_inf=tb.Tools.t_inside(tb.Tools.rm_diag(self.ts_res[i].D)),
                                       gc_pvalues=pvalues,
                                       true_coef=tb.Tools.t_inside(tb.Tools.rm_diag(self.ts_res[i].A)),
                                       p=self.ts_res[i].p)
        tmp_df.iloc[::2, :] = [[package, "Point-Biserial", pb[l][0][0], pb[l][0][1], l + 1, v] for l in
                               np.arange(len(pb))]
        tmp_df.iloc[1::2, :] = [[package, "Pearson", pears[l][0][0], pears[l][0][1], l + 1, v] for l in
                                np.arange(len(pears))]
        return tmp_df

    def plot_acc(self, verbose=True, chosen_lag=None):
        chosen_lag = self.ts.p if chosen_lag is None else chosen_lag
        num = self.df_acc["Package"].nunique()
        fig, axes = plt.subplots(nrows=num, figsize=(8, 5 * num))
        ax = dict(zip(self.df_acc["Package"].unique(), axes))
        with sns.axes_style("darkgrid"):
            for n, p in self.df_acc.groupby("Package"):
                tmp_sd = p.iloc[:, [1, 3]].groupby(self.parameter_name).std()
                tmp_mean = p.iloc[:, [1, 3]].groupby(self.parameter_name).mean()
                tmp_max = p.iloc[:, [1, 3]].groupby(self.parameter_name).max()
                tmp_min = p.iloc[:, [1, 3]].groupby(self.parameter_name).min()
                tmp_chosen_lag = p[p["Lag"] == chosen_lag].set_index(self.parameter_name)["Accuracy"]
                sd = [tmp_mean["Accuracy"].to_numpy() - tmp_sd["Accuracy"].to_numpy(),
                      tmp_mean["Accuracy"].to_numpy() + tmp_sd["Accuracy"].to_numpy()]
                ax[n].plot(tmp_chosen_lag, color=sns.color_palette("tab10")[0], lw=2, marker="o",
                           label=f"Results [lag(s) = 1:{chosen_lag}]")
                ax[n].plot(tmp_mean, color=sns.color_palette("tab10")[1], label=r"$\mu$", marker="d", ls="dashed",
                           alpha=0.2)
                ax[n].fill_between(x=self.parameter_values, y1=sd[1], y2=sd[0], alpha=0.1, label=r"$\mu \pm \sigma$")
                ax[n].plot(tmp_min, color=sns.color_palette("tab10")[3], label="min", ls="dashdot", alpha=0.5)
                ax[n].plot(tmp_max, color=sns.color_palette("tab10")[3], label="max", ls="dotted", alpha=0.5)
                ax[n].legend(loc="best")
                ax[n].set_xlabel(self.parameter_name)
                ax[n].set_ylabel("Accuracy")
                ax[n].set_ylim(-0.05, 1.05)
                ax[n].set_xticks(self.parameter_values)
                ax[n].set_title(f"Accuracy of {n}")
            plt.tight_layout()
            plt.savefig(
                    fname=f"{self.fpath}/sens_accuracy_{self.parameter_name}_{str(datetime.datetime.now())}.pdf",
                    format="pdf")
            if verbose:
                plt.show()
            else:
                plt.close(fig)

    def plot_corr(self, verbose=True, sig_lvl=0.05):
        num = self.df_corr["Package"].nunique()
        fig, axes = plt.subplots(nrows=num, ncols=2, figsize=(12, 5 * num))
        ax = dict(zip(self.df_corr.groupby(["Package", "Type"]).groups.keys(), axes.flat))
        with sns.axes_style("darkgrid"):
            for n, d in self.df_corr.groupby(["Package", "Type"]):
                for i, l in d.groupby("Lag"):
                    tmp_corr = l["Corr"].to_numpy()
                    tmp_p = l["p"].to_numpy()
                    ax[n].plot(self.parameter_values, tmp_corr, marker=self._marker[int(i)], alpha=0.4, ls="--",
                               color="k")
                    tmp_sig = np.where(tmp_p < sig_lvl, tmp_corr, np.nan)
                    if np.nansum(tmp_sig) != 0:
                        ax[n].plot(self.parameter_values, tmp_sig, marker=self._marker[int(i)],
                                   label=f"lags 1:{int(i)} signif. at {sig_lvl:.0%}")
                        ax[n].legend(loc="best")
                ax[n].set_xlabel(self.parameter_name)
                ax[n].set_ylabel("Correlation")
                ax[n].set_ylim(-1.05, 1.05)
                ax[n].set_xticks(self.parameter_values)
                ax[n].set_title(f"{n[1]} Correlation of {n[0]}")
        plt.tight_layout()
        plt.savefig(fname=f"{self.fpath}/sens_corr_{self.parameter_name}_{str(datetime.datetime.now())}.pdf",
                    format="pdf")
        if verbose:
            plt.show()
        else:
            plt.close(fig)

    def plot_partial_comp(self, verbose=True):
        for i, o in enumerate(self.granger_objects):
            o.plot_partial_comp(add=f"({self.parameter_name} = {self.parameter_values[i]:.4f}",
                                cwd=tb.Tools.create_dir_if_missing(dir=f"{self.fpath}/single", return_dir=True),
                                verbose=verbose)

    def plot_multi_comp(self, verbose=True):
        for i, o in enumerate(self.granger_objects):
            o.plot_multi_comp(add=f"{self.parameter_name} = {self.parameter_values[i]:.4f}",
                              cwd=tb.Tools.create_dir_if_missing(dir=f"{self.fpath}/single", return_dir=True),
                              verbose=verbose)

    def print_and_save_panel_data(self, save=True, verbose=True):
        tmp_dict_to_print = {
            "Panel with correlation data": tabulate(self.df_corr, headers="keys", tablefmt="psql"),
            "Panel with accuracy data": tabulate(self.df_acc, headers="keys", tablefmt="psql")
        }
        tmp_dict_latex = {
            "panel_with_correlation_data_as_LaTex": tabulate(self.df_corr, headers="keys", tablefmt="latex"),
            "Panel_with_accuracy_data_as_LaTex": tabulate(self.df_acc, headers="keys", tablefmt="latex")
        }

        for k, v in tmp_dict_to_print.items():
            print(k)
            print(v)
            if save:
                self.__save(v, f"{k}.txt", self.fpath)

        if save:
            for k, v in tmp_dict_latex.items():
                self.__save(v, f"{k}.txt", self.fpath)

    def __save(self, data, filename, path):
        with open(f"{path}/{filename}", "a") as f:
            print(data, file=f)

    def print_corr_summary(self, type="Pearson", verbose=True):
        tmp_corr = self.df_corr.copy()
        tmp_corr = tmp_corr.fillna(0)

        mean = tmp_corr.loc[tmp_corr["Type"] == type].groupby(["Package"])["Corr"].mean()
        std = tmp_corr.loc[tmp_corr["Type"] == type].groupby(["Package"])["Corr"].std()
        min = tmp_corr.loc[tmp_corr["Type"] == type].groupby(["Package"])["Corr"].min()
        max = tmp_corr.loc[tmp_corr["Type"] == type].groupby(["Package"])["Corr"].max()

        mean_all = tmp_corr.loc[tmp_corr["Type"] == type].iloc[:, tmp_corr.columns == "Corr"].mean()
        std_all = tmp_corr.loc[tmp_corr["Type"] == type].iloc[:, tmp_corr.columns == "Corr"].std()
        min_all = tmp_corr.loc[tmp_corr["Type"] == type].iloc[:, tmp_corr.columns == "Corr"].min()
        max_all = tmp_corr.loc[tmp_corr["Type"] == type].iloc[:, tmp_corr.columns == "Corr"].max()

        cols = ["Mean", "Std.", "Min", "Max"]
        all = pd.DataFrame(pd.concat([mean_all, std_all, min_all, max_all], axis=1, keys=cols), columns=cols)
        all.index = ["All packages"]
        tmp_ = pd.concat([mean, std, min, max], axis=1, keys=cols)
        df = tmp_.append(all)

        fig, ax = plt.subplots()
        sns.boxenplot(x="Package", y="Corr", data=tmp_corr.loc[tmp_corr["Type"] == type], ax=ax)
        ax.axhline(y=all.iloc[0, 0], label="Overall mean", ls="dotted", alpha=0.5)
        plt.ticklabel_format(style='plain', axis='y')
        plt.tight_layout()
        plt.savefig(f"{self.fpath}/{type}_sens_corr_boxplot.pdf", format="pdf")
        if verbose:
            fig.show()
        plt.close(fig)

        tab = tabulate(df, headers="keys", tablefmt="latex")
        with open(f"{self.fpath}/{type}_sens_corr_summary.txt", "a") as f:
            print(tab, file=f)
