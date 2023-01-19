import numpy
import matplotlib.pyplot as plt
import pandas as pd
from web3 import Web3

data= pd.read_csv("ETHBUSD-1s-2023-01-08.csv")

data.columns = ['open_time','open', 'high', 'low', 'close', 'volume','close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
# print(data.columns)

# data.plot("open_time", "open")
# data.plot("open_time", "volume")

# plt.figure()
# plt.plot(data["open_time"], data["open"])
# plt.plot(data["open_time"], data["volume"])

w3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/87eb40f105da40bfa9b446730d555fd8'))

#w3 = Web3(Web3.WebsocketProvider("wss://mainnet.infura.io/ws/v3/87eb40f105da40bfa9b446730d555fd8"))

print(w3.isConnected())

w3.eth.get_block('latest')

david_hash = "0x004f7c052cb4ee9abc92294615bca6c55445c99d799a359bd554868ebbd30269"
david_tx = w3.eth.get_transaction(david_hash)
# print(david_tx)

david_receipt = w3.eth.get_transaction_receipt(david_hash)
# print(david_receipt)

Ox_proxy_eth_mainnet = w3.toChecksumAddress("0xdef1c0ded9bec7f1a1670819833240f027b25eff")
Ox_topic0 = "0x0f6672f78a59ba8e5e5b5d38df3ebc67f3c792e2c9259b8d97d7f00dd78ba1b3"
starting_block = 16377745
end_block = 16377857

contract = w3.eth.contract(address=Ox_proxy_eth_mainnet, abi=[{"inputs":[],"stateMutability":"nonpayable","type":"constructor"},{"stateMutability":"payable","type":"fallback"},{"inputs":[{"internalType":"bytes4","name":"selector","type":"bytes4"}],"name":"getFunctionImplementation","outputs":[{"internalType":"address","name":"impl","type":"address"}],"stateMutability":"view","type":"function"},{"stateMutability":"payable","type":"receive"}])
print(contract.abi)
contract.events.

Ox_filter = w3.eth.filter({'fromBlock': starting_block, 'toBlock': end_block, "address": Ox_proxy_eth_mainnet, "topics": [Ox_topic0]})
print(Ox_filter.filter_id)
print(w3.eth.get_filter_changes(Ox_filter.filter_id))
# w3.eth.get_logs(Ox_filter)
# , 'fromBlock': starting_block, 'toBlock': end_block
#w3.eth.get_logs(Ox_filter)

# new_block_filter = w3.eth.filter('latest')
# new_block_filter.get_new_entries()