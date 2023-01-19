# %%
import pandas as pd
import etherscanclient
import estimationtools
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np 

# Import data saved after main_etherscan_test.py
# Indded, queries to etherscan sometimes failed due to their high number
# file_name1 = "saved_data/0xProtocol_LimitOrderFilledEvent_from23010903_to23010909"
# file_name2 = "saved_data/0xProtocol_oldRfqOrderFilledEvent_from23010903_to23010909"
# file_name3 = "saved_data/0xProtocol_RfqOrderFilledEvent_from23010903_to23010909"
# file_name4 = "saved_data/0xProtocol_TransformedERC20Event_from23010903_to23010909"
file_name1 = "saved_data/0xProtocol_LimitOrderFilledEvent_from221201_at_00h00m00_to221214_at_00h00m00"
file_name2 = "saved_data/0xProtocol_oldRfqOrderFilledEvent_from221201_at_00h00m00_to221214_at_00h00m00"
file_name3 = "saved_data/0xProtocol_RfqOrderFilledEvent_from221201_at_00h00m00_to221214_at_00h00m00"
file_name4 = "saved_data/0xProtocol_TransformedERC20Event_from221201_at_00h00m00_to221214_at_00h00m00"

# Load data from Binance corresponding to the same data slot as above queries to compute delta
# data_binance= pd.read_csv("saved_data/ETHBUSD-1s-2023-01-09.csv")
filename_binance = "saved_data/ETHBUSD-1s-2022-12.csv"
data_binance = pd.read_csv(filename_binance)

data_binance.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'QuoteAssetVolume','num_trades','taker_buy_base_vol','taker_buy_quote_vol', 'ignore']
data_binance["open_time_in_seconds"] = (data_binance["open_time"].astype("int")/1000).apply(datetime.fromtimestamp)

pd_data_LimitOrderFilledEvent = etherscanclient.load_data(file_name1)
pd_data_oldRfqOrderFilledEvent = etherscanclient.load_data(file_name2)
pd_data_RfqOrderFilledEvent = etherscanclient.load_data(file_name3)
pd_data_TransformedERC20Event = etherscanclient.load_data(file_name4)


parameter_estimator = estimationtools.VariableEstimateForModel(pd_data_RfqOrderFilledEvent, data_binance)



# Sort transaction depending whether it is WETH->USDC or USDC->WETH
aggreg_data_USDCtoWETH, aggreg_data_WETHtoUSDC = parameter_estimator.sort_data_by_swap_direction()
# print(aggreg_data_USDCtoWETH)

# Compute delta for each direction of swap and returns pandas
delta_WETHtoUSDC, delta_USDCtoWETH = parameter_estimator.compute_delta()

# Print some stats about deltas
parameter_estimator.show_general_stats()

# Plot results: reference data (here Binance) and scatter plot is the transactions that occured on 0x protocol. Higher color saturation correspond to higher delta compared to reference (here Binance)
ax = parameter_estimator.aggreg_data_WETHtoUSDC.plot.scatter(x="TX_timestamps_date", y="Computed_price", c=parameter_estimator.aggreg_data_WETHtoUSDC["Delta in price"], cmap="Greens", title="0x protocol Rfq orders vs binance prices")

parameter_estimator.aggreg_data_USDCtoWETH.plot.scatter(x="TX_timestamps_date", y="Price_converted", c=-parameter_estimator.aggreg_data_USDCtoWETH["Delta in price"], cmap="Reds", ax=ax)

x_custom = data_binance[data_binance["open_time_in_seconds"].between(datetime(2022, 12, 1, 0), datetime(2022, 12, 14, 0))]

x_custom.plot(x="open_time_in_seconds", y="open", alpha=0.2, ax=ax)
ax.set_xlabel("Time")
ax.set_ylabel("WETHUSDC")
ax.legend(["0x WETH to USDC", "0x USDC to WETH", "Binance ETH BUSD"])


# Compute CDF values to prepare for the sigmoid fit
value = parameter_estimator.get_CDF_values(parameter_estimator.aggreg_data_WETHtoUSDC["Delta in bps"], 150)

# Sigmoid fit
popt, pcov = parameter_estimator.fit_sigmoid_to_CDF(aggreg_data_WETHtoUSDC["Delta in bps"])

# Plot an histogram of the whole TX data for one direction of swap and shows the CDF 
fig, ax = plt.subplots()
ax.hist(parameter_estimator.aggreg_data_WETHtoUSDC["Delta in bps"], bins=150, density=True, alpha=0.5, label="Histo", color="tab:orange")
ax.set_ylabel("PDF")
ax.yaxis.label.set_color("tab:orange")
ax.spines["right"].set_edgecolor("tab:orange")
ax.tick_params(axis='y', colors="tab:orange")

ax2 = ax.twinx()
lns1, = ax2.plot(value.index, value.values, label="CDF")
ax2.set_ylabel("CDF")
ax2.set_title("CDF and PDF, range 20k-30k")
ax2.set_xlabel("Delta in bps")
2
ax2.yaxis.label.set_color(lns1.get_color())
ax2.spines["right"].set_edgecolor(lns1.get_color())
ax2.tick_params(axis='y', colors=lns1.get_color())


ax2.plot(value.index, estimationtools.sigmoid(value.index, *popt), "--", label="Fit CDF")
ax2.legend()

plt.show()

alpha, beta = parameter_estimator.compute_alpha_beta(aggreg_data_WETHtoUSDC["Delta in bps"])
print(f"\u03B1 is {alpha:.2f}, \u03B2 is {beta:.2f} bps^{-1}")

# % Make a map of alpha and beta parameters for differents sizes. You can removed some TX size if there is not enought TX. In any case, if sigmoid fit gives an error, it will be removed from plot.

list_outlier = [value*10**3 for value in []] # 50, 130, 140, 150, 160
list_pd = [(i+1)*10**4 for i in range(10)]

ax, alpha, beta, nb_TX_per_bin_size, valid_bins, list_errors = parameter_estimator.plot_alpha_beta_map(aggreg_data_WETHtoUSDC, list_pd, list_outlier)


# % Simple command to get a panda with alpha, beta, lambda and the size for which the parameters were computed.
calibration_matrix = parameter_estimator.get_demand_parameters(aggreg_data_WETHtoUSDC, list_pd, list_errors)



# %% TODOS

# TODO: need to correct the price computation, either wrong from the source, inverse 10**X or else
# For this reason, I'll be working with RfqOrderFilledEvent data for the time being.

# TODO: better way to separate prices than <1 ou >1 car si les tokens sont proches de 1...

# TODO: récupérer les prix de USDC/BUSD

# TODO: create a function get_parameters

# TODO: calculer lambda

# TODO: custom colormap for transaction number like <10 --> red <50 red, etc... <1000 --> dark green

# TODO: utiliser les paramètres dans la sandbox

# TODO: faire histogram et calcul de alpha et beta jusqu'au max des TX s'il y en a plus de 10 par exemple ? 

# TODO: write in the read me for this file what exactly is expected as a head for the ref (binance here) and for the aggregator data.