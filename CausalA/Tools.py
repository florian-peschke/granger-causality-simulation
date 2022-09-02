import datetime
import os

import numpy as np
import pandas as pd
import scipy.stats as scs
import seaborn as sns
import sklearn.metrics as skm
import statsmodels.formula.api as reg
import statsmodels.tsa.tsatools as sm
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
from statsmodels.stats.anova import anova_lm

sns.set(rc={"figure.dpi": 125, 'savefig.dpi': 300})


class Tools:
    @staticmethod
    def corr_test(gc_pvalues, true_inf, true_coef, package, decay, cwd, verbose=True, sig_lvl=0.05, partial=False):
        # checks the correlation between the coefficients of the VAR model and Granger-test results
        p = gc_pvalues.shape[0]
        act_coeffs_p_val_corr, p_val_causal_infl_corr = Tools.calc_corr(gc_pvalues=gc_pvalues, true_inf=true_inf,
                                                                        true_coef=true_coef, p=p, partial=partial)
        with sns.axes_style("darkgrid"):
            sns.set_color_codes()
            fg = plt.figure(figsize=(12, 5), tight_layout=True)
            ax = fg.gca()
            x_ticks = np.arange(1, p + 1)
            plt.plot(x_ticks, decay, color=sns.color_palette("tab10")[0], marker="^",
                     label="Strength decrease of the coefficients",
                     ls="dotted", alpha=0.75, lw=1)
            plt.plot(x_ticks, p_val_causal_infl_corr[:, 0, 0], marker="+",
                     label=f"Point biserial correlation (true causal influence, GC p-values)",
                     ls="--", alpha=0.4, color=sns.color_palette("tab10")[1], lw=1.5)
            plt.plot(x_ticks, act_coeffs_p_val_corr[:, 0, 0], marker="d",
                     label=f"Pearson correlation (true coefficients, GC p-values)",
                     ls="--", alpha=0.4, color=sns.color_palette("tab10")[2], lw=1.5)
            plt.plot(x_ticks,
                     np.where(p_val_causal_infl_corr[:, 0, 1] < sig_lvl, p_val_causal_infl_corr[:, 0, 0], np.nan),
                     color=sns.color_palette("tab10")[1],
                     marker="P",
                     ls="solid", lw=2)
            plt.plot(x_ticks,
                     np.where(act_coeffs_p_val_corr[:, 0, 1] < sig_lvl, act_coeffs_p_val_corr[:, 0, 0], np.nan),
                     color=sns.color_palette("tab10")[2],
                     marker="D",
                     ls="solid", lw=2)
            at = AnchoredText(
                    f"A saturated marker/solid line indicates significance at {sig_lvl:.0%}",
                    prop=dict(size=10),
                    frameon=True,
                    loc="lower right")
            at.patch.set_alpha(0.5)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            ax.add_artist(at)
            plt.xticks(x_ticks)
            plt.xlabel("Lags")
            plt.ylabel("(Correlation)")
            plt.ylim(-1.05, 1.05)
            plt.title(
                    f"Correlation btw. the true influence [binary] and coefficients [continuous], and the GC p-values ["
                    f"continuous] (package: {package})")
            ax.legend(loc="best")
            plt.savefig(fname=f"{cwd}/corr_{package}_{str(datetime.datetime.now())}.pdf", format="pdf")
            if verbose:
                plt.show()
            else:
                plt.close(fg)

            return act_coeffs_p_val_corr, p_val_causal_infl_corr

    @staticmethod
    def t_inside(mat):
        # transposes each matrix of a tensor
        return np.array([np.transpose(i) for i in mat])

    @staticmethod
    def rm_diag(mat):
        # removes the main diogonal
        assert mat.ndim == 3
        r = mat.shape[1]
        c = mat.shape[2]
        tmp_c_sel = [np.delete(np.arange(c), i) for i in np.arange(c)]
        tmp_r_sel = np.repeat(np.arange(r), repeats=r - 1).reshape(-1, r - 1)
        return mat[:, tmp_r_sel, tmp_c_sel]

    @staticmethod
    def deep_inplace_replace(a_to_replace, attribute, dtype=None):
        """
        Takes a tensor of objects and replaces the objects with a specific attribute provided by those objects.

        :param dtype: datatype of elements in the array that will be returned
        :type dtype: str
        :param a_to_replace: array that contains objects
        :type a_to_replace: 3D array that contains objects
        :param attribute: attribute of the objects in a_to_replace that are called
        :type attribute: str
        :return: returns a new array with the same shape but replaces the initial values with the returned values of the
        called attribute
        :rtype: 3D array
        """
        new_array = np.empty(a_to_replace.shape, dtype=dtype)
        # deep iteration over all axis
        for d in np.arange(new_array.shape[0]):
            for r in np.arange(new_array.shape[1]):
                for c in np.arange(new_array.shape[2]):
                    new_array[d][r][c] = getattr(a_to_replace[d][r][c], attribute)
        return new_array

    @staticmethod
    def calc_corr(gc_pvalues, true_inf, true_coef, p, partial=False):
        '''
        Calculates the correlation coefficients (see appendix of the paper for a detailed explanation)
        '''
        p_val_caus_infl_corr = np.zeros((p, 1, 2))
        act_coeffs_p_val_corr = np.zeros((p, 1, 2))
        for l in np.arange(p):
            if partial:
                act_coeffs_p_val_corr[l] = np.array(scs.pearsonr(true_coef[l].flatten(),
                                                                 gc_pvalues[l].flatten()))
                p_val_caus_infl_corr[l] = np.array(scs.pointbiserialr(true_inf[l].flatten(),
                                                                      gc_pvalues[l].flatten()))
                continue
            if true_coef.shape[1] != gc_pvalues.shape[1]:
                gc_pvalue_ext = np.tile(np.tile(gc_pvalues[l], true_coef.shape[2] - 1).reshape(-1, true_coef.shape[2]),
                                        l + 1)
                act_coeffs_p_val_corr[l] = np.array(scs.pearsonr(true_coef[:(l + 1)].flatten(),
                                                                 gc_pvalue_ext.flatten()))
                p_val_caus_infl_corr[l] = np.array(scs.pointbiserialr(true_inf[:(l + 1)].flatten(),
                                                                      gc_pvalue_ext.flatten()))
            else:
                act_coeffs_p_val_corr[l] = np.array(scs.pearsonr(true_coef[:(l + 1)].flatten(),
                                                                 np.tile(gc_pvalues[l], l + 1).flatten()))
                p_val_caus_infl_corr[l] = np.array(scs.pointbiserialr(true_inf[:(l + 1)].flatten(),
                                                                      np.tile(gc_pvalues[l], l + 1).flatten()))
        return act_coeffs_p_val_corr, p_val_caus_infl_corr

    @staticmethod
    def confussion_matrices(predictions, actual):
        confusion_matrices = list()
        for lag, est in enumerate(predictions):
            confusion_matrices.append(
                    skm.confusion_matrix(y_true=actual[:(lag + 1)].flatten().astype(int), y_pred=est.flatten(),
                                         labels=[0, 1]))
        return confusion_matrices

    @staticmethod
    def plot_true_detections(confusion_matrices, test_type, package, decay, cwd, verbose=True):
        with sns.axes_style("darkgrid"):
            cf_res = np.array(
                    [[np.divide(np.trace(i), i.sum()), 1 - np.divide(np.trace(i), i.sum())] for i in
                     confusion_matrices])
            fg = plt.figure(figsize=(10, 5))
            x_ticks = np.arange(1, len(confusion_matrices) + 1)
            plt.plot(x_ticks, decay, color=sns.color_palette("tab10")[0], marker="^",
                     label="Strength decrease of the coefficients",
                     ls="dotted")
            plt.plot(x_ticks, cf_res[:][:, 0], color=sns.color_palette("tab10")[1],
                     marker="o", label="Rate of detected causal influences")
            plt.title(f"Rate of truly detected causal influences by lags (package: {package} | test-type: {test_type})")
            plt.xlabel("Lags")
            plt.ylabel("(Accuracy)")
            plt.ylim(-0.05, 1.05)
            plt.legend(loc="best")
            plt.xticks(x_ticks)
            plt.tight_layout()
            plt.savefig(fname=f"{cwd}/true_det_{package}_{str(datetime.datetime.now())}.pdf", format="pdf")
            if verbose:
                plt.show()
            else:
                plt.close(fg)

    @staticmethod
    def plot_cm(confusion_matrices, package_name, cwd, verbose=False):
        p = len(confusion_matrices)
        with sns.axes_style("ticks"):
            fig, axes = plt.subplots(nrows=p, figsize=(10, 5 * len(confusion_matrices)), ncols=2, tight_layout=True)
            fig.suptitle(f"Confusion matrices of <{package_name}>", y=.99)
            for i, ax in enumerate(axes.flat):
                r = np.floor(i / 2).astype(int)
                labels = [0, 1]
                if i % 2 == 0:
                    sns.heatmap(confusion_matrices[r], annot=True,
                                cmap="Blues", ax=ax)
                else:
                    sns.heatmap(confusion_matrices[r] / np.sum(confusion_matrices[r]), annot=True, fmt=".2%",
                                cmap="Blues", ax=ax)
                ax.set_xlabel("\nPredicted Values")
                ax.set_ylabel("Actual Values ")
                ax.xaxis.set_ticklabels(labels)
                ax.yaxis.set_ticklabels(labels)
                ax.set_title(f"lags 1:{r + 1}")
            plt.savefig(fname=f"{cwd}/confusion_mat_{package_name}_{str(datetime.datetime.now())}.pdf",
                        format="pdf")
            if verbose:
                plt.show()
            else:
                plt.close(fig)

    @staticmethod
    def rm_el(array, element):
        tmp_arr = array.copy()
        tmp_arr.remove(element)
        return tmp_arr

    @staticmethod
    def gen_values_from_beta_dist(alpha, beta, value, pos):
        assert 0 < value < 1
        beta = scs.beta(alpha, beta)
        p = beta.cdf(value)
        pb = np.arange(1, pos + 1) / (pos + 1) * p
        return beta.ppf(pb)

    @staticmethod
    def partial_granger_causality(x: pd.DataFrame, y: pd.DataFrame, p, out=None):
        """
        Tests whether L(x,5) => y
        :param x: the time series, that might Granger cause with respect to the coefficient of its pth lag
        :param y: time series that might be Granger caused by L(x,5)
        :param p: lag value
        :param out:
        :returns: p-value of the F-test whether L(x,5) => y
        """
        # generates the autoregressive terms
        y_lags = [f"y{i}" for i in np.arange(1, p + 1)]
        # generates the regressors
        regressors = y_lags + [f"x{p}"]
        # labels for the dataset
        data_labels = ["y"] + regressors
        # combine all values to one dataset
        val = np.hstack(
                [y.iloc[p:].to_numpy().reshape(-1, 1), sm.lagmat(y, p)[p:, :], x.iloc[p:].to_numpy().reshape(-1, 1)])
        data = pd.DataFrame(val, columns=data_labels)
        # restricted model
        r_formula = f"y ~  {' + '.join(str(r) for r in y_lags)}"
        r_model = reg.ols(r_formula, data).fit()
        # unrestricted model
        ur_formula = f"y ~  {' + '.join(str(r) for r in regressors)}"
        ur_model = reg.ols(ur_formula, data).fit()
        # standard F-test
        f_test = anova_lm(r_model, ur_model)
        # returns the p-value of the F-test
        if out is None:
            return f_test.iloc[-1, -1]  # ur_model.pvalues[-1]
        # assigns the p-value of the F-test to the variable out
        out = f_test.iloc[-1, -1]

    @staticmethod
    def create_dir_if_missing(dir, return_dir=False):
        if not os.path.exists(dir):
            os.makedirs(dir)
        if return_dir:
            return dir
