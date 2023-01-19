#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:22:28 2023
Tools to estimate parameters for price model
@author: Shigami
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tabulate import tabulate
from datetime import timedelta
from itertools import compress

def sigmoid(bps, alpha, beta, b, L0=1):
    """
    Sigmoid function where 
    L0 is the curve maximum value
    alpha is the sigmoid's midpoint
    beta is the logistic growth rate
    b is the offset/bias in y
    """
    y = L0 / (1 + np.exp(alpha+beta*bps)) + b
    return (y)


class TransactionData():
    """
    Represent a data frame containing information necessary to estimate model parameters.
    """

    def __init__(self, tx_object):
        pass


class VariableEstimateForModel():
    """
    Class to make all computation about parameters alpha, beta, lambda.
    """
    def __init__(self, pandas_aggreg_data, pandas_ref_data):
        self.aggreg_data = pandas_aggreg_data.sort_values(by=['BlockNumber'], ignore_index=True)
        self.ref_data = pandas_ref_data

    def sort_data_by_swap_direction(self):
        """"
        Sort data from aggreg_data depending on the direction of the swap.
        """
        aggreg_data_USDCtoWETH = self.aggreg_data[self.aggreg_data["Computed_price"]<1]
        aggreg_data_USDCtoWETH = aggreg_data_USDCtoWETH.reset_index()

        aggreg_data_USDCtoWETH = aggreg_data_USDCtoWETH.assign(Price_converted = 1/aggreg_data_USDCtoWETH.Computed_price)
        aggreg_data_WETHtoUSDC = self.aggreg_data[self.aggreg_data["Computed_price"]>1]
        aggreg_data_WETHtoUSDC = aggreg_data_WETHtoUSDC.reset_index()
        
        self.aggreg_data_USDCtoWETH = aggreg_data_USDCtoWETH
        self.aggreg_data_WETHtoUSDC = aggreg_data_WETHtoUSDC
        return self.aggreg_data_USDCtoWETH, self.aggreg_data_WETHtoUSDC
        
    def find_ref_corresponding_to_aggreg(self, aggreg_data):
        """
        Returns subset of the aggreg DataFrame with match timestamp (1s ref, max blocktimestamp)
        """
        ref_data_overlap = pd.DataFrame()
        for i in aggreg_data["TX_timestamps_date"]:
            ref_data_overlap = pd.concat([ref_data_overlap, self.ref_data[self.ref_data["open_time_in_seconds"].eq(i)]])

        return ref_data_overlap.reset_index()

    def compute_delta(self):
        """
        Compute price difference between aggreg and reference.
        """
        corresponding_ref_data_USDCtoWETH = self.find_ref_corresponding_to_aggreg(self.aggreg_data_USDCtoWETH)
        delta_USDCtoWETH =  self.aggreg_data_USDCtoWETH["Price_converted"] - corresponding_ref_data_USDCtoWETH["open"]

        corresponding_ref_data_WETHtoUSDC = self.find_ref_corresponding_to_aggreg(self.aggreg_data_WETHtoUSDC)
        delta_WETHtoUSDC = corresponding_ref_data_WETHtoUSDC["open"] - self.aggreg_data_WETHtoUSDC["Computed_price"] 

        self.aggreg_data_WETHtoUSDC["Delta in price"] = delta_WETHtoUSDC
        self.aggreg_data_USDCtoWETH["Delta in price"] = delta_USDCtoWETH
        
        self.aggreg_data_WETHtoUSDC["Delta in profit"] = self.compute_delta_in_profits(delta_WETHtoUSDC, self.aggreg_data_WETHtoUSDC["convertedTakerTokenFilledAmount"])

        self.aggreg_data_USDCtoWETH["Delta in profit"] = self.compute_delta_in_profits(delta_USDCtoWETH, self.aggreg_data_USDCtoWETH["convertedMakerTokenFilledAmount"])

        self.aggreg_data_WETHtoUSDC["Delta in bps"] = self.aggreg_data_WETHtoUSDC["Delta in price"]/self.aggreg_data_WETHtoUSDC["Computed_price"]*10**4

        self.aggreg_data_USDCtoWETH["Delta in bps"] = self.aggreg_data_USDCtoWETH["Delta in price"]/self.aggreg_data_USDCtoWETH["Computed_price"]*10**4

        return self.aggreg_data_WETHtoUSDC["Delta in price"], self.aggreg_data_USDCtoWETH["Delta in price"]

    def compute_delta_in_profits(self, delta, amount):
        return delta*amount
        
    def show_general_stats(self):
        print(tabulate(
            [['Mean delta', np.mean(self.aggreg_data_WETHtoUSDC['Delta in price']), 'USDC'],
            ['Median delta', self.aggreg_data_WETHtoUSDC['Delta in price'].median(), 'USDC'],
            ['Mean delta in profit', np.mean(self.aggreg_data_WETHtoUSDC['Delta in profit']), 'USDC'],
            ['Median delta in profit', self.aggreg_data_WETHtoUSDC['Delta in profit'].median(), 'USDC'],
            ['Mean delta in bps', np.mean(self.aggreg_data_WETHtoUSDC['Delta in bps']), 'USDC'],
            ['Median delta in bps', self.aggreg_data_WETHtoUSDC['Delta in bps'].median(), 'USDC'],
            ['Total profits', np.sum(self.aggreg_data_WETHtoUSDC['Delta in profit']), 'USDC']],
            headers=['Stats type', 'Value', 'Currency']))

    def get_CDF_values(self, pd_column, bins_nb=150):
        # if pd_column in self.keys():
        values = (pd_column.pipe(lambda s: pd.Series(np.histogram(s, range=(min(pd_column), max(pd_column)), bins=bins_nb)))
                .pipe(lambda s: pd.Series(s[0], index=s[1][:-1]))
                .pipe(lambda s: pd.Series(np.cumsum(s[::-1])[::-1]/np.sum(s))))
        # else: 
        #     raise(Exception(f"Given key {pd_column} does not exist"))
        return values

    def fit_sigmoid_to_CDF(self, pd_colomn, bins_nb=150):
        # p0_exp = [-0.3, 0.4, 0, 1]
        values = self.get_CDF_values(pd_colomn, bins_nb=bins_nb)
        popt, pcov = curve_fit(sigmoid, values.index, values.values, method='dogbox')

        return popt, pcov

    def compute_alpha_beta(self, pd_colomn):
        popt, pcov = self.fit_sigmoid_to_CDF(pd_colomn)
        alpha = popt[0]
        beta = popt[1]

        return alpha, beta

    def split_colomn_by_size(self, pd_table, bin_edges):
        pd_by_size = []
        edge_left = 0
        print(f"Largest TX = {max(pd_table['convertedMakerTokenFilledAmount']):.2f} USDC")
        for edge_right in bin_edges:
            filtered_pd = pd_table[pd_table["convertedMakerTokenFilledAmount"].between(edge_left, edge_right)]
            filtered_pd = filtered_pd.reset_index()
            pd_by_size.append(filtered_pd)
            edge_left = edge_right
        
        return pd_by_size

    def compute_alpha_beta_by_size(self, pd_table, bin_edges_size, exempt_bins=[]):
        """
        Compute alpha and beta values for each given size. It is useeful for the model calibration.
        """
        pd_by_size = self.split_colomn_by_size(pd_table, bin_edges_size)
        nb_TX_per_bin_size = []
        alpha = []
        beta = []
        error_sigmoiderror_sigmoid_fit_bin_fit = []
        valid_bins = []

        curated_bin_edges_size = [edge_size for edge_size in bin_edges_size if edge_size not in exempt_bins]
        curated_pd_by_size = [pd_by_size_i for i, pd_by_size_i in enumerate(pd_by_size) if bin_edges_size[i] not in exempt_bins]

        for i, pd_one_size in enumerate(curated_pd_by_size):
            try:
                alpha_tmp, beta_tmp = self.compute_alpha_beta(pd_one_size["Delta in bps"])
                # Append value only if success of compute_alpha_beta
                valid_bins.append(curated_bin_edges_size[i])
                alpha.append(alpha_tmp)
                beta.append(beta_tmp)
                nb_TX_per_bin_size.append(len(pd_one_size))
            except RuntimeError:
                print(f"For size {curated_bin_edges_size[i]}: RuntimeError: Optimal parameters not found: The maximum number of function evaluations is exceeded.")
                error_sigmoiderror_sigmoid_fit_bin_fit.append(curated_bin_edges_size[i])

        return alpha, beta, nb_TX_per_bin_size, valid_bins, error_sigmoiderror_sigmoid_fit_bin_fit           

    def plot_alpha_beta_map(self, pd_table, bin_edges_size, outlier_list=[]):
        """
        Make a map of beta = f(alpha). The color of the dot represents the number of TX for that bin ammount.
        """
        alpha, beta, nb_TX_per_bin_size, valid_bins, list_errors = self.compute_alpha_beta_by_size(pd_table, bin_edges_size, exempt_bins=outlier_list)

        fig, ax = plt.subplots()
        cmap_custom = (mpl.colors.ListedColormap(['red', 'orange', 'tab:green', 'DarkGreen'])
        )
        bounds = [0, 10, 50, 100, max(nb_TX_per_bin_size)]
        norm = mpl.colors.BoundaryNorm(bounds, cmap_custom.N)

        ax.scatter(alpha, beta, c=nb_TX_per_bin_size, cmap=cmap_custom, norm=norm)
        sm = plt.cm.ScalarMappable(cmap=cmap_custom, norm=norm) #, spacing='uniform'
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=bounds)
        cbar.set_label('# of TX', rotation=270)
        ax.set_xlabel("\u03B1")
        ax.set_ylabel("\u03B2 bps^-1")

        list_errors.extend(outlier_list[:])
        for i, txt in enumerate(valid_bins):
            ax.annotate(f"{txt/10**3:.0f}k", (alpha[i], beta[i]), rotation=45)
        ax.set_title(f"Removed outlier/fit errors {[f'{i/10**3:.0f}k' for i in list_errors]}")

        return ax, alpha, beta, nb_TX_per_bin_size, valid_bins, list_errors

    def split_pd_by_day(self, pd_table):
        """
        Separate the pandas into a list where each element is a pandas with data of 1 day
        """
        def daterange(date1, date2):
            dates = []
            for n in range(int ((date2 - date1).days)+3):
                date_tmp = date1 + timedelta(n)
                dates.append(pd.to_datetime(date_tmp.date()))

            return dates

        pd_table_day_by_day = []

        date_list = daterange(pd_table["Query start date"].iloc[0], pd_table["Query end date"].iloc[0])

        for i, date_ii in enumerate(date_list[:-1]):
            date_i = date_list[i+1]
            pd_table_day_by_day.append(pd_table[pd_table["TX_timestamps_date"].between(date_ii, date_i)])

        return pd_table_day_by_day, date_list

    def get_number_TX_bySize_byDay(self, pd_table, bin_edges, exempt_bins=[]):
        pd_size_tmp = self.split_colomn_by_size(pd_table, bin_edges)
        pd_size = list(compress(pd_size_tmp, [x not in exempt_bins for x in bin_edges]))

        lambda_list = []
        for size_i in pd_size:
            pd_size_date, date_list = self.split_pd_by_day(size_i)

            lambda_size_date_list = []
            for i in range(len(date_list[:-2])):
                lambda_size_date = len(pd_size_date[i])
                lambda_size_date_list.append(lambda_size_date)

            lambda_list.append(lambda_size_date_list)

        return lambda_list

    def compute_lambda(self, pd_table, bin_edges, exempt_bins=[]):
        lambda_list = self.get_number_TX_bySize_byDay(pd_table, bin_edges, exempt_bins=exempt_bins)

        return np.max(lambda_list, axis=1)

    def get_demand_parameters(self, pd_table, bin_edges_size, exempt_bins=[]):
        alpha, beta, _, valid_bins, _ = self.compute_alpha_beta_by_size(pd_table, bin_edges_size, exempt_bins=exempt_bins)
        lambda_list = self.compute_lambda(pd_table, bin_edges_size, exempt_bins=exempt_bins)
        result_dict = {
            "alpha": alpha,
            "beta": beta,
            "lambda": lambda_list,
            "TX_size": valid_bins
        }
        pd_results = pd.DataFrame(result_dict)

        return pd_results
