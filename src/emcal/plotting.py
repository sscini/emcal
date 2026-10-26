import numpy as np
import math
import pandas as pd
import os
import matplotlib.ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections.abc import Iterable

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from .analysis import *
from .analysis import is_job_like  # explicit (robust if analysis adds __all__)
from .case_studies import get_cs_class_from_val

import warnings

np.warnings = warnings


class Plotter:
    """
    The base class for per-job/diagnostic plotting functions. Multi-job/cross-method/
    benchmark plotting (all-methods comparisons, NLS/derivative-free baselines, cross-case-
    study bar charts) has moved to the archive repo.

    Methods
    --------------
    __init__(analyzer, save_figs = False): Constructor method
    plot_hyperparameters(job, title = None): Plots hyperparameters vs BO Iter for all methods
    plot_parameters(job, z_choice, title = None): Plots parameter sets vs BO Iter for all methods
    __plot_2D_general(data, data_names, data_true, y_label, title, log_data): Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots
    custom_format(x, pos): Custom format for 10x notation
    __get_z_plot_names_hms(z_choice): Returns the names of the z values for the plot
    plot_gp_fit(z_choice, log_data = False, title = None): Plots comparison of y_sim, GP_mean, GP_stdev, and EI at the best runs
    __scale_z_data(data, z_choice): Scales the z data based on the z choice
    __set_ylab_from_z(z_choice): Returns the y label based on the z choice
    __get_data_to_bo_iter_term(data): Returns the data up to the termination of the BO iteration
    __save_fig(save_path): Saves the figure to the save path
    __create_subplots(subplots_needed, sharex = False, sharey = 'none'): Creates subplots based on the number of subplots needed
    __set_subplot_details(ax, x_space, data_df_j, x_label, y_label, title): Sets the details of the subplot
    __set_plot_titles(fig, title, x_label, y_label): Sets the title and labels of the plot
    """

    # Class variables and attributes

    def __init__(self, analyzer, save_figs=False):
        """
        Parameters
        ----------
        analyzer: RunAnalysis
            An instance of the RunAnalysis class
        save_figs: bool, default False
            Save figures to file if True.

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Asserts
        assert isinstance(save_figs, bool), "save_figs must be boolean"
        assert isinstance(analyzer, RunAnalysis), "analyzer must be RunAnalysis"

        # Constructor method
        self.analyzer = analyzer
        self.save_figs = save_figs
        self.cmap = "YlOrRd_r"
        self.xbins = 5
        self.ybins = 5
        self.title_fntsz = 24
        self.other_fntsz = 24
        self.method_names = [
            "Conventional",
            "Log Conventional",
            "Independence",
            "Log Independence",
            "Sparse Grid",
            "Monte Carlo",
            "E[SSE]",
        ]

    def plot_hyperparameters(self, job, title=None):
        """
        Plots hyperparameters vs BO Iter for all methods

        Parameters
        -----------
        job: signac.job.Job
            The job to analyze
        title: str or None, default None
            Title of plot
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        data, data_names, data_true, sp_data = self.analyzer.hyperparameter_trajectories(job)
        y_label = "Value"
        title = "Hyperparameter Values"
        fig = self.__plot_2D_general(data, data_names, data_true, y_label, title, False)
        # save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "hyperparams")
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def plot_parameters(self, job, z_choice, title=None):
        """
        Plots parameter sets vs BO Iter for all methods for a give z_choice (min_sse, sse, or acq)

        Parameters
        -----------
        job: signac.job.Job
            The job to analyze
        z_choice: str
            One of "min_sse", "sse", or "acq". The values that will be plotted
        title: str or None, default None
            Title of plot
        """
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(z_choice, str), "z_choice must be a string"
        assert z_choice in [
            "min_sse",
            "sse",
            "acq",
        ], "z_choice must be one of 'min_sse', 'sse', or 'acq'"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        data, data_names, data_true, sp_data = self.analyzer.parameter_trajectories(
            job, z_choice
        )
        be, be_theta = self.analyzer.best_error(job)
        GPBO_method_val = sp_data["meth_name_val"]
        # Create label based on method #
        meth_label = self.method_names[GPBO_method_val - 1]
        y_label = "Parameter Values"

        # Get number of iterations to add at the beginning by grabbing the length of theta_true
        cs_class = get_cs_class_from_val(sp_data["cs_name_val"])
        num_params = len(cs_class.idcs_to_consider)
        num_train_points = sp_data["num_theta_multiplier"] * num_params

        if title != None:
            title = title
        else:
            title = meth_label + " Parameter Values"

        fig = self.__plot_2D_general(
            data,
            data_names,
            data_true,
            y_label,
            title,
            False,
            num_train_points,
            be_theta,
        )
        # save or show figure
        if self.save_figs:
            save_path_to = os.path.join(job.fn(""), "line_plots", "params_" + z_choice)
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

    def __plot_2D_general(
        self,
        data,
        data_names,
        data_true,
        y_label,
        title,
        log_data,
        num_train_points=0,
        be=None,
    ):
        """
        Plots 2D values of the same data type (ei, sse, min sse) on multiple subplots

        Parameters
        -----------
        data:np.ndarray (n_runs x n_iters x n_params)
            Array of data from bo workflow runs
        data_names: list(str)
            List of data names
        data_true: list/ndarray(float/int) or None,
            The true/reference values of each parameter
        y_label: str
            The y label of the plot
        title: str
            The title of the plot
        log_data: bool
            Plots data on natural log scale if True

        Returns
        --------
        fig: plt.figure, The figure object
        """

        # Number of subplots is number of parameters for 2D plots (which will be the last spot of the shape parameter)
        subplots_needed = data.shape[-1]
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True
        )
        # Print the title and labels as appropriate
        self.__set_plot_titles(fig, title, None, None)

        # Loop over different hyperparameters (number of subplots)
        for i, ax in enumerate(axes.flatten()):
            # Only plot data if axis is visible
            if i < subplots_needed:
                # The index of the data is i, and one data type is in the last row of the data
                one_data_type = data[:, :, i]

                # Loop over all runs
                for j in range(one_data_type.shape[0]):
                    # Create label based on run #
                    label = "Run: " + str(j + 1)
                    data_df_j = self.__get_data_to_bo_iter_term(one_data_type[j])

                    # Define x axis
                    bo_len = len(data_df_j) + num_train_points
                    bo_space = np.linspace(1, bo_len, bo_len)

                    # Set appropriate notation
                    if abs(np.max(data_df_j)) >= 1e3 or abs(np.min(data_df_j)) <= 1e-3:
                        fmt = matplotlib.ticker.FuncFormatter(self.custom_format)
                        ax.yaxis.set_major_formatter(fmt)

                    # Plot data
                    if log_data == True:
                        data_df_j = np.log(data_df_j)

                    # duplicate the first value num_train_points times to show the number of training points
                    if num_train_points > 0:
                        data_train = be[j, i] if be is not None else data_df_j[0]
                        data_df_j = np.concatenate(
                            (np.full(num_train_points, data_train), data_df_j)
                        )
                        x_label = "Loss Evaluations"
                    else:
                        x_label = "BO Iterations"

                    ax.step(bo_space, data_df_j, label=label)

                    # Plot true value if applicable
                    if data_true is not None and j == one_data_type.shape[0] - 1:
                        ax.axhline(
                            y=list(data_true.values())[i],
                            color="red",
                            linestyle="--",
                            label="True Value",
                        )

                    # Set plot details
                    title = r"$" + data_names[i] + "$"
                    self.__set_subplot_details(
                        ax, bo_space, data_df_j, None, None, title
                    )

                if not log_data and data_true is None:
                    ax.set_yscale("log")

            # Add legends and handles from last subplot that is visible
            if i == subplots_needed - 1:
                handles, labels = axes[0, -1].get_legend_handles_labels()

        for axs in axes[-1]:
            axs.set_xlabel(x_label, fontsize=self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(y_label, fontsize=self.other_fntsz)

        # Plots legend and title
        plt.tight_layout()
        fig.legend(
            handles,
            labels,
            loc="center left",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(1.0, 0.60),
            borderaxespad=0,
        )

        return fig

    def custom_format(self, x, pos):
        """
        Custom format for 10x notation

        Parameters
        -----------
        x: float
            The value to format
        pos: int
            The position of the value

        Returns
        --------
        formatted_str: str
            The formatted value

        Notes
        ------
        Returns 0 if x is 0 and formats the value using scientific notation otherwise
        """
        if x == 0:
            return "0"
        formatted = "{:2.2e}".format(x)  # Format the value using scientific notation
        mantissa, exponent = formatted.split("e")
        return r"${} \times 10^{{{}}}$".format(mantissa, int(exponent))

    # def add_training_iters(self, data):

    def __get_z_plot_names_hms(self, z_choices, sim_sse_var_ei):
        """
        Returns the z data and title for the heat map plots

        Parameters
        -----------
        z_choices: str
            One of "sse_sim", "sse_mean", "sse_var", or "acq". The values that will be plotted
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            Tuple of the data from the self.analyzer.gp_heat_map_data() method

        Returns
        --------
        all_z_data: list(np.ndarray)
            List of z data for each objective to plot
        all_z_titles: list(str)
            Mathematical titles for each objective to plot
        all_z_titles_pre: list(str)
            Titles for each objective to plot

        Raises
        ------
        ValueError
            If z_choice is not one of "sse_sim", "sse_mean", "sse_var", or "acq"
        """
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        all_z_data = []
        all_z_titles = []
        all_z_titles_pre = []
        # Find z based on z_choice
        # Fix me: Heat Maps always use just theta: Only the bar labels need to change
        for z_choice in z_choices:
            if "sse_sim" == z_choice:
                title = r"$\mathscr{L}(\mathbf{\theta})$"
                # title = r"$g(\mathbf{\theta})$"
                all_z_data.append(sse_sim)
                all_z_titles.append(title)
                all_z_titles_pre.append("SSE Loss Function, ")
            elif "sse_mean" == z_choice:
                title = r"$\tilde{\mathscr{L}}(\mathbf{\theta})$"
                # title = r"$\tilde{g}(\mathbf{\theta})$"
                all_z_data.append(sse_mean)
                all_z_titles.append(title)
                all_z_titles_pre.append("(Predicted) SSE Loss Function, ")
            elif "sse_var" == z_choice:
                all_z_data.append(sse_var)
                all_z_titles_pre.append("Predicted Variance, ")
                all_z_titles.append(
                    r"$\mathbf{\sigma}^2_{\tilde{\mathscr{L}}(\mathbf{\theta})}$"
                )
            elif "acq" == z_choice:
                all_z_data.append(ei)
                all_z_titles.append(r"$\Xi(\mathbf{\theta})$")
                all_z_titles_pre.append("Acquisition Function, ")
            else:
                raise ValueError("choice must contain 'sim', 'mean', 'var', or 'acq'")
        return all_z_data, all_z_titles, all_z_titles_pre

    def plot_gp_fit(
        self,
        job,
        run_num,
        bo_iter,
        pair,
        z_choices,
        levels=7,
        log_data=False,
        title=None,
    ):
        """
        Plots comparison of y_sim, GP_mean, and GP_stdev
        Parameters
        ----------
        job: signac.job.Job
            The job to analyze
        run_num: int
            The run number to analyze
        bo_iter: int
            The bo iteration to analyze
        pair: int
            The pair of data parameters. pair 0 is the 1st pair
        z_choices: str, list(str),
            One of "sse_sim", "sse_mean", "sse_var", or "acq". The values that will be plotted
        levels: int, list(int), or None
            Number of zbins to skip when drawing contour lines
        log_data: bool, default False
            Plots data on natural log scale if True
        title: str or None, default None
            Title of plot

        Returns
        -------
        plt.show(), A heat map of test_mesh and z

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
            If meshgrids are not the correct shape
            If there are not enough levels for the number of subplots
        Warning
            If log_data is True and minimum values to plot is less than or equal to 0

        Notes
        -------
        For this function, each objective function value is its own subplot. Each plot must be generated separately for each method.
        """
        # Assert Statements
        assert is_job_like(job), "job must be a signac job or a JobContext"
        assert isinstance(run_num, int), "run_num must be an integer"
        assert isinstance(bo_iter, int), "bo_iter must be an integer"
        assert isinstance(pair, int), "pair must be an integer"
        assert isinstance(log_data, bool), "log_data must be boolean"
        assert isinstance(title, str) or title is None, "title must be a string or None"
        assert isinstance(
            z_choices, (Iterable, str)
        ), "z_choices must be Iterable or str"
        if isinstance(z_choices, str):
            z_choices = [z_choices]
        for z_choice in z_choices:
            assert z_choice in [
                "sse_sim",
                "sse_mean",
                "sse_var",
                "acq",
            ], "z_choices elements must be 'sse_sim', 'sse_mean', 'sse_var', or 'acq'"

        assert isinstance(levels, (int, list)), "levels must be an int or list"
        # Define plot levels
        if isinstance(levels, int):
            levels = [levels] * len(z_choices)
        else:
            levels = levels
        assert len(levels) == len(
            z_choices
        ), "levels must be int or have the same length as z_choices"
        # Get all data for subplots needed
        get_ei = True if "acq" in z_choices else False
        analysis_list = self.analyzer.gp_heat_map_data(
            job, run_num, bo_iter, pair, get_ei=get_ei
        )
        sim_sse_var_ei, test_mesh, param_info_dict, sp_data = analysis_list
        # Get method value from json file
        GPBO_method_val = sp_data["meth_name_val"]
        # Set correct values based on propagation of errors for gp
        sim_sse_var_ei = self.__scale_z_data(sim_sse_var_ei, sp_data, log_data)
        theta_true = param_info_dict["true"]
        theta_opt = param_info_dict["min_sse"]
        theta_next = param_info_dict["opt_acq"]
        train_theta = param_info_dict["train"]
        plot_axis_names = param_info_dict["names"]
        idcs_to_plot = param_info_dict["idcs"]

        # Assert sattements
        # Get x and y data from test_mesh
        xx, yy = test_mesh  # NxN, NxN
        assert xx.shape == yy.shape, "Test_mesh must be 2 NxN arrays"

        # Make figures and define number of subplots based on number of files (different methods)
        subplots_needed = len(z_choices)
        fig, axes, num_subplots, plot_mapping = self.__create_subplots(
            subplots_needed, sharex=True, sharey=True
        )

        # Find z based on z_choice
        all_z_data, all_z_titles, all_z_titles_pre = self.__get_z_plot_names_hms(
            z_choices, sim_sse_var_ei
        )

        sse_cond = None

        # Loop over number of subplots
        for i, ax in enumerate(axes.flatten()):
            if i < subplots_needed:
                # Get data for z_choice
                z = all_z_data[i]
                need_unscale = False

                # Unlog scale the data if vmin is 0 and log_data = True
                if np.min(z) == -np.inf or np.isnan(np.min(z)) or np.min(z) == 0:
                    need_unscale = True
                    if log_data:
                        warnings.warn(
                            "Cannot plot log scaled data! Reverting to original"
                        )
                        z = np.exp(all_z_data[i])

                # Create normalization
                vmin = np.nanmin(z)
                vmax = np.nanmax(z)
                # If all z data are the same, nudge vmin/vmax apart so contourf's levels are
                # strictly increasing. The nudge must scale with the data's own magnitude: a
                # fixed 1e-14 is far below float64 precision for anything but values near
                # zero (e.g. vmin=vmax=337 +/- 1e-14 rounds right back to 337), which silently
                # left vmin == vmax true and crashed downstream with "Contour levels must be
                # increasing" -- reproduced by a GP with an exactly-flat predictive mean.
                # Nearly flat counts too: on macOS a "flat" surface can vary by one ulp.
                eps = max(abs(vmax), 1.0) * 1e-6
                if vmax - vmin < eps:
                    vmin -= eps
                    vmax += eps

                # Check if data scales 3 orders of magnitude
                mag_diff = (
                    int(math.log10(abs(vmax)) - math.log10(abs(vmin))) > 2.0
                    if vmin > 0
                    else False
                )

                if need_unscale == False and log_data:
                    title2 = "log(" + all_z_titles[i] + ")"
                else:
                    title2 = all_z_titles[i]

                condition = log_data or vmin < 0 or not mag_diff
                if "sse" in z_choices[i]:
                    if sse_cond is None:
                        sse_cond = condition
                    condition = sse_cond
                # Choose an appropriate colormap and scaling based on vmin, vmax, and log_data
                # If not using log data, vmin > 0, and the data scales 3 orders+ of magnitude use log10 to view plots
                if condition:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.linspace(vmin, vmax, levels[i])
                    nticks = cbar_ticks

                else:
                    norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=False)
                    cbar_ticks = np.logspace(
                        np.log10(vmin), np.log10(vmax), levels[i]
                    )  # Set 5 equally spaced ticks
                    # Get log10 scale bounds
                    min_power = np.floor(
                        np.log10(vmin)
                    )  # Round down the logarithm to get the closest power of 10
                    max_power = np.ceil(
                        np.log10(vmax)
                    )  # Round up the logarithm to get the closest power of 10
                    # Create the ticks at powers of 10 within this range
                    nticks = np.logspace(
                        min_power, max_power, int(max_power - min_power + 1)
                    )

                # Create a colormap and colorbar normalization for each subplot
                cs_fig = ax.contourf(
                    xx,
                    yy,
                    z,
                    levels=cbar_ticks,
                    # tick_positions=nticks,
                    cmap=plt.colormaps[self.cmap],
                    norm=norm,
                )

                # Create a line contour for each colormap
                if levels is not None:
                    num_levels = len(cbar_ticks)
                    indices = np.linspace(
                        0, len(cs_fig.levels) - 1, num_levels, dtype=int
                    )
                    selected_levels = cs_fig.levels[indices]
                    cs2_fig = ax.contour(
                        cs_fig,
                        levels=selected_levels,
                        colors="k",
                        alpha=0.7,
                        linestyles="dashed",
                        linewidths=3,
                        norm=norm,
                    )

                # plot min obj, max ei, true and training param values as appropriate
                if theta_true is not None:
                    ax.scatter(
                        theta_true[idcs_to_plot[0]],
                        theta_true[idcs_to_plot[1]],
                        color="blue",
                        label="True",
                        s=200,
                        marker=(5, 1),
                        zorder=2,
                    )
                if train_theta is not None:
                    ax.scatter(
                        train_theta[:, idcs_to_plot[0]],
                        train_theta[:, idcs_to_plot[1]],
                        color="green",
                        s=100,
                        label="Train",
                        marker="x",
                        zorder=1,
                    )
                if theta_next is not None:
                    ax.scatter(
                        theta_next[idcs_to_plot[0]],
                        theta_next[idcs_to_plot[1]],
                        color="black",
                        s=175,
                        label="acq_value",
                        marker="^",
                        zorder=3,
                    )
                if theta_opt is not None:
                    ax.scatter(
                        theta_opt[idcs_to_plot[0]],
                        theta_opt[idcs_to_plot[1]],
                        color="darkmagenta",
                        s=160,
                        label="Min Obj",
                        marker=".",
                        edgecolor="magenta",
                        linewidth=0.7,
                        zorder=4,
                    )

                # Set plot details
                self.__set_subplot_details(ax, xx, yy, None, None, all_z_titles[i])
                if sp_data["cs_name_val"] in [16, 17]:
                    ax.ticklabel_format(
                        style="scientific", axis="both", scilimits=(-2, 2)
                    )

                # Use a custom formatter for the colorbar
                fmt = matplotlib.ticker.FuncFormatter(self.custom_format)

                divider1 = make_axes_locatable(ax)
                cax1 = divider1.append_axes("right", size="5%", pad="6%")
                cbar = fig.colorbar(
                    cs_fig,
                    ax=ax,
                    cax=cax1,
                    use_gridspec=True,
                    ticks=cbar_ticks,
                )
                cbar.ax.yaxis.set_major_formatter(fmt)
                cbar.ax.tick_params(labelsize=int(self.other_fntsz / 2))

        # Get legend information and make colorbar on last plot
        handles, labels = axes[0, 0].get_legend_handles_labels()

        # Print the title
        if title is None:
            title = self.method_names[GPBO_method_val - 1]

        # For case studies 16 and 17, change the parameter names to be the correct ones from calc_y_fxns
        if sp_data["cs_name_val"] in [16, 17]:
            plot_axis_names = tuple(
                "tau_{12}" if name == "theta_1" else "tau_{21}"
                for name in plot_axis_names
            )

        # Print the title and labels as appropriate
        # Define x and y labels
        if "theta" in plot_axis_names[0] or "tau" in plot_axis_names[0]:
            xlabel = r"$\mathbf{" + "\\" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + "\\" + plot_axis_names[1] + "}$"
        else:
            xlabel = r"$\mathbf{" + plot_axis_names[0] + "}$"
            ylabel = r"$\mathbf{" + plot_axis_names[1] + "}$"

        for axs in axes[-1]:
            axs.set_xlabel(xlabel, fontsize=self.other_fntsz)

        for axs in axes[:, 0]:
            axs.set_ylabel(ylabel, fontsize=self.other_fntsz)

        self.__set_plot_titles(fig, title, None, None)

        # Plots legend and title
        fig.legend(
            handles,
            labels,
            loc="upper right",
            fontsize=self.other_fntsz,
            bbox_to_anchor=(-0.02, 1),
            borderaxespad=0,
        )

        plt.tight_layout()

        # save or show figure
        if self.save_figs:
            z_choices_sort = sorted(
                z_choices,
                key=lambda x: ("sse_sim", "sse_mean", "sse_var", "acq").index(x),
            )
            z_choices_str = "_".join(map(str, z_choices_sort))
            title_str = title.replace(" ", "_").lower()
            save_path_dir = os.path.join(
                job.fn(""),
                "heat_maps",
                title_str,
                plot_axis_names[0] + "-" + plot_axis_names[1],
                z_choices_str,
            )
            save_path_to = os.path.join(
                save_path_dir, "run_" + str(run_num) + "_" + "iter_" + str(bo_iter)
            )
            self.__save_fig(save_path_to)
        else:
            plt.show()
            plt.close()

        return plt.show()

    def __scale_z_data(self, sim_sse_var_ei, sp_data, log_data):
        """
        Scales the objective (sse_sim, sse_gp, sse_var, or acq_func) data based on the method and log_data

        Parameters
        -----------
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            Tuple of the data from the self.analyzer.gp_heat_map_data() method
        sp_data: dict
            Dictionary of the data from the json file
        log_data: bool
            Plots data on natural log scale if True

        Returns
        --------
        sim_sse_var_ei: tuple(np.ndarray, len=4) or tuple(np.ndarray, np.ndarray, np.ndarray, None)
            tuple of the data from the analysis with correct scaling for plots
        """
        sse_sim, sse_mean, sse_var, ei = sim_sse_var_ei
        # Get log or unlogged data values
        if log_data == False:
            # Change sse sim, mean, and stdev to not log for 1B
            if sp_data["meth_name_val"] in [2]:
                # SSE variance is var*(e^((log(sse)))^2
                sse_mean = np.exp(sse_mean)
                sse_var = sse_var * sse_mean**2
                sse_sim = np.exp(sse_sim)

        # If getting log values
        else:
            # Get log data from 1A, 2A, 2B, 2C, and 2D
            if not sp_data["meth_name_val"] in [2]:
                # SSE Variance is var/sse**2
                sse_var = sse_var / sse_mean**2
                sse_mean = np.log(sse_mean)
                sse_sim = np.log(sse_sim)

        sim_sse_var_ei = [sse_sim, sse_mean, sse_var, ei]
        return sim_sse_var_ei

    def __set_ylab_from_z(self, z_choice):
        """
        Sets the y label based on the z_choice

        Parameters
        -----------
        z_choice: str
            One of "sse", "min_sse", or "acq"

        Returns
        --------
        y_label: str
            The y label for the plot
        """

        if self.analyzer.mode == "gp":
            label_g = "\\tilde{\mathscr{L}}(\mathbf{"
            label_a = "(Predicted) SSE Loss Function, "
        else:
            label_g = "\mathscr{L}(\mathbf{"
            label_a = "SSE Loss Function, "
        if "sse" == z_choice:
            theta = "\\theta}^o" if self.analyzer.mode != "acq" else "\\theta^*}"
            y_label = label_g + theta + ")"
        if "min_sse" == z_choice:
            theta = "\\theta}^{\prime}" + ")"
            y_label = label_g + theta
        if "acq" == z_choice:
            label_a = "Acquisition Function, "
            y_label = "\Xi(\mathbf{\\theta^*})"
        final_label = label_a + r"$" + y_label + "$"
        return final_label

    def __get_data_to_bo_iter_term(self, data_all_iters):
        """
        Gets non-zero data for plotting from data array

        Parameters
        -----------
        data_all_iters: np.ndarray
            Data from all iterations

        Returns
        --------
        data_df_j: np.ndarray
            Data that is not numerically 0
        """
        # Remove elements that are numerically 0
        data_df_run = pd.DataFrame(data=data_all_iters)
        data_df_j = data_df_run.loc[(abs(data_df_run) > 1e-14).any(axis=1), 0]
        data_df_i = data_df_run.loc[:, 0]  # Used to be data_df_i
        # Ensure we have at least 2 elements to plot
        if len(data_df_j) < 2:
            data_df_j = data_df_i[
                0 : int(len(data_df_j) + 2)
            ]  # +2 for stopping criteria + 1 to include last point

        return data_df_j

    def __save_fig(self, save_path, ext="png", close=True):
        """Save a figure from pyplot.
        Parameters
        ----------
        save_path : string
            The path (and filename, without the extension) to save the
            figure to.
        ext : string (default='png')
            The file extension. This must be supported by the active
            matplotlib backend (see matplotlib.backends module).  Most
            backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
        close : boolean (default=True)
            Whether to close the figure after saving.  If you want to save
            the figure multiple times (e.g., to multiple formats), you
            should NOT close it in between saves or you will have to
            re-plot it.
        """

        # Extract the directory and filename from the given path
        directory = os.path.split(save_path)[0]
        filename = "%s.%s" % (os.path.split(save_path)[1], ext)
        if directory == "":
            directory = "."

        # If the directory does not exist, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        # The final path to save to
        savepath = os.path.join(directory, filename)

        # Actually save the figure
        plt.savefig(savepath, dpi=300, bbox_inches="tight")

        # Close it
        if close:
            plt.close()

    def __create_subplots(
        self,
        num_subplots,
        sharex="row",
        sharey="none",
        threeD=False,
        x_size=6,
        y_size=6,
    ):
        """
        Creates Subplots based on the amount of data

        Parameters
        ----------
        num_subplots: int
            Total number of needed subplots
        sharex: str, default "row"
            sharex values for subplots
        sharey: str, default "none"
            sharey value for subplots

        Returns
        -------
        fig: matplotlib.figure
            The matplotlib figure object
        axes: matplotlib.axes.Axes
            2D array of axes
        total_ax_num: int
            The number of axes generated total
        plot_mapping: dict
            Dictionary mapping plot number to axes

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value

        """

        assert num_subplots >= 1, "Number of subplots must be at least 1"
        assert isinstance(num_subplots, int), "Num subplots must be int"
        # Make figures and define number of subplots
        # If you are making more than one figure, sharex is always true
        if num_subplots == 1:
            sharex = True

        # Make enough rows and columns and get close to equal number of each
        row_num = int(np.floor(np.sqrt(num_subplots)))
        col_num = int(np.ceil(num_subplots / row_num))
        assert (
            row_num * col_num >= num_subplots
        ), "row * col numbers must be at least equal to number of graphs"
        total_ax_num = row_num * col_num

        # Creat subplots
        gridspec_kw = {"wspace": 0.4, "hspace": 0.2}
        if threeD:
            subplot_kw = {"projection": "3d"}
        else:
            subplot_kw = {}
        fig, axes = plt.subplots(
            row_num,
            col_num,
            figsize=(col_num * x_size, row_num * y_size),
            squeeze=False,
            sharex=sharex,
            sharey=sharey,
            subplot_kw=subplot_kw,
        )

        # Turn off unused axes
        for i, axs in enumerate(axes.flatten()):
            if i >= num_subplots:
                axs.axis("off")

        # Make plot mapping to map an axes to an iterable value
        plot_mapping = {}
        for i in range(row_num):
            for j in range(col_num):
                plot_number = i * col_num + j
                plot_mapping[plot_number] = (i, j)

        return fig, axes, total_ax_num, plot_mapping

    def __set_subplot_details(
        self, ax, plot_x, plot_y, xlabel, ylabel, title, plot_z=None, zlabel=None
    ):
        """
        Function for setting plot settings

        Parameters
        ----------
        ax: matplotlib.axes.Axes
            The axes to set the plot settings for
        plot_x: np.ndarray
            The x data for plotting
        plot_y: np.ndarray
            The y data for plotting
        xlabel: str or None
            The label for the x axis
        ylabel: str or None
            The label for the y axis
        title: str or None
            The subplot title

        Returns
        -------
        ax: matplotlib.axes.Axes
            The axes with the plot settings set

        Raises
        ------
        AssertionError
            If any of the required parameters are missing or not of the correct type or value
        """
        # Group inputs by type
        none_str_vars = [title, xlabel, ylabel]
        int_vars = [self.xbins, self.ybins, self.other_fntsz]
        arr_vars = [plot_x, plot_y]

        # Assert Statements
        assert all(
            isinstance(var, str) or var is None for var in none_str_vars
        ), "title, xlabel, and ylabel must be string or None"
        assert all(
            isinstance(var, int) for var in int_vars
        ), "xbins, ybins, and fontsize must be int"
        assert all(
            var > 0 or var is None for var in int_vars
        ), "integer variables must be positive"
        assert all(
            isinstance(var, (np.ndarray, pd.core.series.Series)) or var is None
            for var in arr_vars
        ), "plot_x, plot_y must be np.ndarray or pd.core.series.Series or None"

        # Set title, label, and axes
        if title is not None:
            pad = 6 + 4 * title.count("_")
            ax.set_title(title, fontsize=self.other_fntsz, fontweight="bold", pad=pad)
        if xlabel is not None:
            pad = 4 * xlabel.count("_") + self.other_fntsz * 1.1
            ax.set_xlabel(
                xlabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )
        if ylabel is not None:
            pad = 4 * ylabel.count("_") + self.other_fntsz * 1.1
            ax.set_ylabel(
                ylabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )
        if zlabel is not None:
            pad = 5 * zlabel.count("_") + self.other_fntsz
            ax.set_zlabel(
                zlabel, fontsize=self.other_fntsz, fontweight="bold", labelpad=pad
            )

        # Turn on tick parameters and bin number
        ax.xaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=5)
        ax.yaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=5)
        ax.locator_params(axis="y", nbins=self.ybins)
        ax.locator_params(axis="x", nbins=self.xbins)
        ax.minorticks_on()  # turn on minor ticks
        ax.tick_params(which="minor", direction="in", top=True, right=True)

        # Set a and y bounds and aspect ratio
        if plot_z is None:
            if plot_x is not None and not np.isclose(
                np.min(plot_x), np.max(plot_x), rtol=1e-6
            ):
                ax.set_xlim(left=np.min(plot_x), right=np.max(plot_x))

            if plot_y is not None and abs(np.min(plot_y)) <= 1e-16:
                ax.set_ylim(ymin=1e-16, ymax=np.max(plot_y) * 1.1)

            if plot_y is not None and (np.min(plot_y) == np.max(plot_y) == 0):
                ax.set_ylim(bottom=np.min(plot_y) - 0.05, top=np.max(plot_y) + 0.05)

            ax.set_box_aspect(1)
        else:
            ax.zaxis.set_tick_params(labelsize=self.other_fntsz, direction="in", pad=10)
            ax.locator_params(axis="z", nbins=self.ybins)
            if plot_x is not None and not np.isclose(
                np.min(plot_x), np.max(plot_x), rtol=1e-6
            ):
                ax.set_xlim(left=np.min(plot_x), right=np.max(plot_x))

            if plot_y is not None and not np.isclose(
                np.min(plot_y), np.max(plot_y), rtol=1e-6
            ):
                ax.set_ylim(bottom=np.min(plot_y), top=np.max(plot_y))

            if plot_z is not None:
                max_value = np.maximum(np.max(plot_z), 5)
                ax.set_zlim(zmin=0, zmax=max_value)
            ax.set_box_aspect([1, 1, 1])

        return ax

    def __set_plot_titles(self, fig, title, x_label, y_label):
        """
        Helper function to set plot titles and labels for figures with subplots

        Parameters
        ----------
        fig: matplotlib.figure
            The figure to set the title and labels for
        title: str or None
            The title of the figure
        x_label: str or None
            The x label of the figure
        y_label: str or None
            The y label of the figure
        """
        if self.title_fntsz is not None:
            fig.suptitle(title, weight="bold", fontsize=self.title_fntsz)
        if x_label is not None:
            fig.supxlabel(x_label, fontsize=self.other_fntsz, fontweight="bold")
        if y_label is not None:
            fig.supylabel(y_label, fontsize=self.other_fntsz, fontweight="bold")
        return

