import datetime
import os

import num2words as n2w
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame

from CausalA import Tools as tb

sns.set(rc={"figure.dpi": 125, "savefig.dpi": 300})


class TS_Sim:
    """
    Class to generate the set of time series.
    """

    _are_bp_gen = False
    _are_coeffs_gen = False
    _coeffs_type_alo = ["Only AR", "Only DL", "ADL"]
    # NNI: Nearest-Neighbor increasing strength
    # NND: Nearest-Neighbor decreasing strength
    _coeffs_form_alo = ["NNI", "NND", "Random"]

    def __init__(
        self,
        samples,
        number_ts,
        lags=1,
        number_structural_breaks=0,
        seed=None,
        decay_basis=None,
        decay_velocity=None,
        ts_with_sb=None,
        coef_influence=None,
        coefficient_tensor=None,
        error_mean_tensor=None,
        error_covariance_tensor=None,
        break_points=None,
        trend=None,
        constant=None,
        coeffs_type="Only DL",
        coeffs_form="NNI",
        tril=True,
        alpha=2,
        beta=2,
        decay_strength=1,
        cwd="./",
        scale_var=1,
        scale_cov=0,
        allow_neg_coeffs=False,
    ) -> None:
        """
        If a parameter is optional, it will be randomly generated or calculated from existing values,
        when no explicit values are given.

        :param samples: Number of samples
        :param number_ts: Number of time series
        :param lags: Number of lags
        :param number_structural_breaks: Number of structural breaks
        :param seed: Seed
        :param decay_basis: Decay basis (see appendix in the paper)
        :param decay_velocity: Can be an array of fixed values that is going to be multiplied with the respective
        lag coefficient matrix
        :param ts_with_sb: Array with the 0s and 1s. Where 0 indicates that the time series shows no structural break
        and 1 indicates that the time series shows a structural break
        :param coef_influence: Can be a binary tensor where a 0 element indicates no causal influence and a 1 element
        indicates causal influence
        :param coefficient_tensor: Tensor that includes the coefficients
        :param error_mean_tensor: Tensor that includes mean values of the error terms
        :param error_covariance_tensor: Tensor that includes variance-covariance matrices of the error terms
        :param break_points: Tensor with the specific break points (per time series)
        :param trend: Array with the values of the time trend per time series
        :param constant: Array with the values of the constant per time series
        :param coeffs_type: ["Only AR", "Only DL", "ADL"] => Only AR considers only the autoregressive values; Only
        DL considers only the distributed lag values; ADL considers both
        :param coeffs_form: ["NNI", "NND", "Random"] => The AR value per time series is the reference (see appendix
        in the paper) | NNI: Nearest-Neighbor increasing; NND: Nearest-Neighbor decreasing; Random: completly random
        coefficients that do not depend on the AR value
        :param tril: Wether only the lower triangle of the true causal influence matrix should be used
        :param alpha: Alpha value of the Beta distribution
        :param beta: Beta value of the Beta distribution
        :param decay_strength: Decay strength (see appendix in the paper)
        :param cwd: Current working directory
        :param scale_var: Scaling scalar of the variance that are included in the variance-covariance matrix of
        the error terms
        :param scale_cov: Scaling scalar of the covariance that are included in the variance-covariance matrix of
        the error terms
        :param allow_neg_coeffs: Whether coefficients can be negative or not
        """
        self.summary_data = None
        self.T = samples
        self.p = lags
        self.k = number_ts
        self.nsb = number_structural_breaks
        self.seed = seed
        self.rdg = np.random.default_rng(self.seed)
        self.decay_strength = decay_strength
        self.d = decay_basis if decay_basis is not None else self.rdg.uniform()
        self.cwd = cwd
        self.allow_neg_coeffs = allow_neg_coeffs
        self.scale_var = scale_var
        self.scale_cov = scale_cov
        self.time = str(datetime.datetime.now())
        if decay_velocity is not None:
            self.a = decay_velocity
        else:
            self.__gen_decay(self.d, self.decay_strength)
        self.ts_wsb = ts_with_sb if ts_with_sb is not None else self.rdg.choice(a=[0, 1], size=(self.k, 1))
        self.__set_coef_det(coeffs_type, coeffs_form)
        self.tril = tril
        self.alpha = alpha
        self.beta = beta
        self.D = coef_influence if coef_influence is not None else np.ones((self.p, self.k, self.k))
        if coefficient_tensor is not None:
            self.A = coefficient_tensor
        else:
            self.__gen_coeffs()
        self.init_error_mean_vec = (
            error_mean_tensor if error_mean_tensor is not None else np.zeros(shape=(1, self.k, 1))
        )
        self.__set_error(error_covariance_tensor=error_covariance_tensor)
        self.__set_break_points(break_points)
        self.trend = trend if trend is not None else np.zeros(self.k)
        self.constant = constant if constant is not None else np.zeros(self.k)
        self.__gen_dir()
        self.__gen_ts()

    def __gen_dir(self):
        self.cwdf = f"{self.cwd}/figures"
        self.fpath = f"{self.cwdf}/time_series"
        if not os.path.exists(self.fpath):
            os.makedirs(self.fpath)

    def __gen_decay(self, d, decay_strength):
        self.a = np.array([d / np.power(i, decay_strength) for i in np.arange(1, self.p + 1)])

    def __set_error(self, error_covariance_tensor):
        """
        Generates the mean and variance-covariance tensor of the error terms
        """
        self.err_mean = (
            self.init_error_mean_vec
            if self.nsb == 0
            else np.vstack([self.init_error_mean_vec, np.abs(self.rdg.standard_t(df=1, size=(self.nsb, self.k, 1)))])
        )
        self.err_cov = error_covariance_tensor if error_covariance_tensor is not None else self.__gen_cov()

    def __gen_cov(self) -> ndarray:
        return np.array(
            [
                self.__gen_corr(
                    size=(self.k, self.k), var=np.abs(self.rdg.normal(loc=0, scale=self.scale_var, size=self.k))
                )
                for _ in np.arange(self.nsb + 1)
            ]
        )

    def __gen_corr(self, size, var):
        """
        Creating the positive definitevariance-covariance matrix of the error terms.
        """
        mat = np.multiply(np.tril(self.rdg.normal(size=size)), self.scale_cov)
        np.fill_diagonal(mat, var)
        sigma = mat @ mat.T
        cholesky = np.linalg.cholesky(sigma)
        positive_semi_definite = cholesky @ cholesky.T
        return positive_semi_definite

    def __set_break_points(self, break_points):
        """
        Generates the time points for each time series where it should break.
        """
        if break_points is not None:
            self.pure_bp = break_points
        else:
            self.pure_bp = np.multiply(
                np.array(self.rdg.choice(np.arange(1, self.T), replace=False, size=(self.nsb, self.k))), self.ts_wsb.T
            )

            self._are_bp_gen = True
        self._break_points = self.__wrap_bp(self.pure_bp + self.p)
        self._break_points[:].sort(axis=0)

    def __wrap_bp(self, break_points):
        return np.vstack([np.repeat(self.p, repeats=self.k), break_points, np.repeat(self.T + self.p, repeats=self.k)])[
            :, :, np.newaxis
        ]

    def __set_coef_det(self, coeffs_type, coeffs_form):
        if coeffs_type not in self._coeffs_type_alo:
            raise Warning(
                f"Test type {coeffs_type} is not supported. Supported test types are: {self._coeffs_type_alo}"
            )
        if coeffs_form not in self._coeffs_form_alo:
            raise Warning(
                f"Test type {coeffs_form} is not supported. Supported test types are: {self._coeffs_form_alo}"
            )
        self.coeffs_type = coeffs_type
        self.coeffs_form = coeffs_form

    def __gen_coeffs(self) -> None:
        """
        Generating the main coefficients by incorporating the vanishing/decay rates.
        """
        self.A = np.zeros((self.p, self.k, self.k))
        self.A_base = self.__gen_main_coef_mat()
        self.A_proc = self.__process_coeffs_type(self.A_base)
        if self.tril:
            self.D = np.tril(np.ones(shape=(self.p, self.k, self.k)), k=-1)
        else:
            self.D = self.rdg.choice(a=[0, 1], size=(self.p, self.k, self.k))
        for d in np.arange(self.p):
            signs = (
                self.rdg.choice(a=[-1, 1], size=(self.k, self.k))
                if self.allow_neg_coeffs
                else np.ones((self.k, self.k))
            )
            np.multiply(signs, np.multiply(self.D[d], np.multiply(self.a[d], self.A_proc)), out=self.A[d])
        self._are_coeffs_gen = True

    def __gen_main_coef_mat(self):
        """
        Generating the coefficients by using the Beta distribution (see appendix in the paper).
        """
        ar = self.rdg.uniform(size=self.k)
        if self.coeffs_form == "NNI":
            return np.array(
                [
                    np.concatenate(
                        [
                            tb.Tools.gen_values_from_beta_dist(self.alpha, self.beta, ar[i], i),
                            [ar[i]],
                            tb.Tools.gen_values_from_beta_dist(self.alpha, self.beta, ar[i], self.k - i - 1)[::-1],
                        ]
                    )
                    for i in np.arange(self.k)
                ]
            )
        elif self.coeffs_form == "NND":
            return np.array(
                [
                    np.concatenate(
                        [
                            tb.Tools.gen_values_from_beta_dist(self.alpha, self.beta, ar[i], i)[::-1],
                            [ar[i]],
                            tb.Tools.gen_values_from_beta_dist(self.alpha, self.beta, ar[i], self.k - i - 1),
                        ]
                    )
                    for i in np.arange(self.k)
                ]
            )
        else:
            return self.rdg.uniform(size=(self.k, self.k))

    def __process_coeffs_type(self, coeffs):
        tmp_mat = np.ones((self.k, self.k))
        if self.coeffs_type == "Only AR":
            tmp_mat = np.zeros((self.k, self.k))
            np.fill_diagonal(tmp_mat, np.diag(coeffs))
        elif self.coeffs_type == "Only ADL":
            np.fill_diagonal(tmp_mat, np.zeros(self.k))
        return np.multiply(tmp_mat, coeffs)

    def __gen_sb(self, t) -> ndarray:
        """
        Function to generate structural breaks (individually for each time series).
        """
        d = np.subtract(
            [np.argmax(self._break_points[:, i, :] >= t) for i in np.arange(self._break_points.shape[1])], 1
        )
        return np.array(
            [
                self.rdg.multivariate_normal(mean=self.err_mean[t].flatten(), cov=self.err_cov[t])[i]
                for i, t in enumerate(d)
            ]
        )

    def __matmul_lags(self, t, lag=None) -> ndarray:
        """
        Calculates the current y_t values according to the number of lags and respective coefficients.
        Implements the Sigma sum function of the main equation stated in the paper in a recursive fashion.
        :param t: Current time point
        :param lag: Number of lags
        :return: Array with the calculated values of y_t
        """
        lag = self.p if lag is None else lag
        if lag <= 0:
            return np.zeros(self.k)
        return np.sum([np.matmul(self.A[lag - 1], self.ts_data[:, t - lag]), self.__matmul_lags(t, lag - 1)], axis=0)

    def __gen_ts(self) -> None:
        """
        Data generating process (time series simulation)
        """
        self.ts_data = np.zeros((self.k, self.T + self.p))
        self.ts_names = np.array([f"t{i + 1}" for i in np.arange(self.k)])
        # generating the data for all time series
        # model is based on Lütkepohl (2005) New Introduction to Multiple Time Series Analysis
        for t in np.arange(self.p, self.T + self.p):
            np.add.reduce(
                [self.constant, np.multiply(self.trend, t - self.p), self.__matmul_lags(t), self.__gen_sb(t)],
                out=self.ts_data[:, t],
            )
        self.df_ts = (
            pd.DataFrame(self.ts_data.T, columns=self.ts_names).drop(np.arange(self.p + 1)).reset_index(drop=True)
        )

    def get_ts(self) -> DataFrame:
        return self.df_ts

    def plot_ts(self) -> None:
        plt.close("all")
        with sns.axes_style("darkgrid"):
            fig, axes = plt.subplots(nrows=self.k, figsize=(10, 3 * self.k))
            fig.suptitle("Simulated time series")
            for ts, ax in enumerate(axes.flat):
                legend_elements = [
                    plt.Line2D([0], [0], lw=1, label=f"{n2w.num2words(ts + 1, to='ordinal_num')} time series")
                ]
                self.df_ts.iloc[:, ts].plot(lw=1, ax=ax)
                if self.nsb > 0 and self.ts_wsb[ts] == 1:
                    for b in np.arange(1, self.nsb + 1):
                        ax.axvline(x=self._break_points[b, ts, 0] - self.p, linestyle="--", color="red", alpha=0.25)
                    legend_elements.append(
                        plt.Line2D(
                            [0], [0], linestyle="--", color="red", alpha=0.25, label=f"Breakpoints (n = {self.nsb})"
                        )
                    )
                ax.legend(handles=legend_elements, loc="best")
        plt.xlabel("Time")
        plt.tight_layout()
        plt.savefig(fname=f"{self.fpath}/ts_{self.time}.pdf", format="pdf")
        plt.show()

    def update(self, parameter_name, parameter_value):
        """
        Function to update this object with the new parameter.
        """
        try:
            setattr(self, parameter_name, parameter_value)
        except NameError:
            raise Warning(
                f"The attribute does not exist or has a typo. The following attributes can be accessed: "
                f"{self.__dict__}"
            )
        self.rdg = np.random.default_rng(self.seed)
        self.time = str(datetime.datetime.now())
        if self._are_coeffs_gen:
            self.__gen_decay(self.d, self.decay_strength)
            self.__gen_coeffs()
        if self._are_bp_gen:
            self.__set_error(None)
            self.__set_break_points(None)
        self.__gen_ts()

    def summary(self, save=True):
        """
        Returns all main parameters of the time series simulation.
        """
        self.summary_data = {
            "coefficient tensor": self.A,
            "causal influence tensor": self.D.astype(int),
            "decay basis": self.d,
            "decay strength": self.decay_strength,
            "decay velocity": self.a,
            "number of structural breaks": self.nsb,
            "time series with structural breaks": self.ts_names[self.ts_wsb.astype(bool).flatten()]
            if self.nsb > 0
            else [],
            "break point tensor": self._break_points[1:-1] - self.p,
            "trend": self.trend,
            "constant": self.constant,
            "error covariance tensor": self.err_cov,
            "error mean": self.err_mean,
        }
        for k, v in self.summary_data.items():
            print(f"{k}:")
            with np.printoptions(precision=4, suppress=True):
                print(v)
        if save:
            self.__save_sum()

    def __save_sum(self):
        with open(f"{self.fpath}/ts_{self.time}.txt", "a") as f:
            for k, v in self.summary_data.items():
                print(f"{k}:", file=f)
                with np.printoptions(precision=4, suppress=True):
                    print(v, file=f)
