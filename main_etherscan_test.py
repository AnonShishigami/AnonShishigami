from datetime import datetime
import etherscanclient
from tqdm import tqdm
# import sha3

# API key from etherscan. You can use your own one.
API_KEY_ETHERSCAN = "U94H8S3YWY1YTI1WVI9YGEAM8MG16HT3QA"

# Connects to a etherscan client.
clientAPI_etherscan = etherscanclient.EtherscanClient(API_KEY_ETHERSCAN)

current_eth_price = clientAPI_etherscan.get_current_eth_price()
print(f"\nCurrent price of ETH is {current_eth_price}")

# Get ETH <-> USDC swaps event from 0x
ADDRESS_0x_PROXY_ETH_MAINNET = "0xdef1c0ded9bec7f1a1670819833240f027b25eff" # 0x aggregator contract
Ox_TOPICS_0_1 = "0x0f6672f78a59ba8e5e5b5d38df3ebc67f3c792e2c9259b8d97d7f00dd78ba1b3" # TransformedERC20
Ox_TOPICS_0_2 = "0xac75f773e3a92f1a02b12134d65e1f47f8a14eabe4eaf1e24624918e6a8b269f"
Ox_TOPICS_0_3 = "0x829fa99d94dc4636925b38632e625736a614c154d55006b7ab6bea979c210c32" # RfqOrderFilled
Ox_TOPICS_0_4 = "0xab614d2b738543c0ea21f56347cf696a3a0c42a7cbec3212a5ca22a4dcff2124" # LimitOrderFilled

# Choose a date. I chose this one because there was some volatility.
t0_sync = datetime(2022, 12, 1, 0)
t1_sync = datetime(2022, 12, 14, 0)

# Converts date to block number on ETH mainnet.
block_number_t0 = clientAPI_etherscan.get_block_number_by_timestamp(t0_sync)
block_number_t1 = clientAPI_etherscan.get_block_number_by_timestamp(t1_sync)
#block_number_t0 = 16382471
#block_number_t1 = 16382591
print(
    f"\nStarting block={block_number_t0}, end block={block_number_t1}, number of blocks={block_number_t1-block_number_t0}")

Ox_events_data_topic_1 = clientAPI_etherscan.get_event_log_by_address_filtered_topics(
    ADDRESS_0x_PROXY_ETH_MAINNET, Ox_TOPICS_0_1,
    fromBlock=block_number_t0, toBlock=block_number_t1)

print(f"\nNumber of TransformedERC20 detected: {len(Ox_events_data_topic_1)}\n")

Ox_events_data_topic_2 = clientAPI_etherscan.get_event_log_by_address_filtered_topics(
    ADDRESS_0x_PROXY_ETH_MAINNET, Ox_TOPICS_0_2,
    fromBlock=block_number_t0, toBlock=block_number_t1)



print(f"Number of old RfqOrderFilled detected: {len(Ox_events_data_topic_2)}\n")

Ox_events_data_topic_3 = clientAPI_etherscan.get_event_log_by_address_filtered_topics(
    ADDRESS_0x_PROXY_ETH_MAINNET, Ox_TOPICS_0_3,
    fromBlock=block_number_t0, toBlock=block_number_t1)

print(f"Number of RfqOrderFilled detected: {len(Ox_events_data_topic_3)}\n")

Ox_events_data_topic_4 = clientAPI_etherscan.get_event_log_by_address_filtered_topics(
    ADDRESS_0x_PROXY_ETH_MAINNET, Ox_TOPICS_0_4,
    fromBlock=block_number_t0, toBlock=block_number_t1)

print(f"Number of LimitOrderFilled detected: {len(Ox_events_data_topic_4)}\n")


decoded_data = Ox_events_data_topic_3[0].decode_event_logs_manual_parser(Ox_TOPICS_0_3)
# print(decoded_data)
# print(Ox_events_data_topic_3[0].tx_hash)

# print(decoded_data["takerTokenFilledAmount"]*10**-18)
# print(decoded_data["makerTokenFilledAmount"]*10**-6)



# TransformedERC20 Event
Ox_event_1_list_tmp = [etherscanclient.OxProtocolEvent(Ox_events_data_topic_1[i]) for i in range(len(Ox_events_data_topic_1))]
Ox_event_1_list = etherscanclient.OxProtocolClientList(Ox_event_1_list_tmp)

# print("\nTopics0_1 Exchange ratio price:")
# print(*Ox_event_1_list.get_exchange_ratio_list(), sep="\n")

# print(f"\ntx hash list :")
# print(*Ox_event_1_list.get_list_tx_hash(), sep="\n")

# old RfqOrderFilled Event maybe ?
Ox_event_2_list_tmp = [etherscanclient.OxProtocolEvent(Ox_events_data_topic_2[i]) for i in range(len(Ox_events_data_topic_2))]
Ox_event_2_list = etherscanclient.OxProtocolClientList(Ox_event_2_list_tmp)

# print("\nTopics0_2 Exchange ratio price:")
# print(*Ox_event_2_list.get_exchange_ratio_list(), sep="\n")

# print(f"\ntx hash list :")
# print(*Ox_event_2_list.get_list_tx_hash(), sep="\n")

# RfqOrderFilled Event
Ox_event_3_list_tmp = [etherscanclient.OxProtocolEvent(Ox_events_data_topic_3[i]) for i in range(len(Ox_events_data_topic_3))]
Ox_event_3_list = etherscanclient.OxProtocolClientList(Ox_event_3_list_tmp)


# print("\nTopics0_3 Exchange ratio price:")
# print(*Ox_event_3_list.get_exchange_ratio_list(), sep="\n")

# print(f"\ntx hash list :")
# print(*Ox_event_3_list.get_list_tx_hash(), sep="\n")

# LimitOrderFilled Event
Ox_event_4_list_tmp = [etherscanclient.OxProtocolEvent(Ox_events_data_topic_4[i]) for i in range(len(Ox_events_data_topic_4))]
Ox_event_4_list = etherscanclient.OxProtocolClientList(Ox_event_4_list_tmp)

# print("\nTopics0_4 Exchange ratio price:")
# print(*Ox_event_4_list.get_exchange_ratio_list(), sep="\n")

# print(f"\ntx hash list :")
# print(*Ox_event_4_list.get_list_tx_hash(), sep="\n")

# Make sure hash correspond to topics0
name3 = b'RfqOrderFilled(bytes32,address,address,address,address,uint128,uint128,bytes32)'
name1 = b'TransformedERC20(address,address,address,uint256,uint256)'
name4 = b'LimitOrderFilled(bytes32,address,address,address,address,address,uint128,uint128,uint128,uint256,bytes32)'

# etherscanclient.compute_event_hash(name1)
# etherscanclient.compute_event_hash(name3)
# etherscanclient.compute_event_hash(name4)


# %% Select swap USDC <> WETH

Ox_event_3_list_tmp[0].block_number
data = Ox_event_3_list_tmp[0].decoded_data

WETH_MAKER_CONTRACT = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".lower()
USDC_CIRCLE_CONTRACT = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48".lower()
maker_taker_addresses = [WETH_MAKER_CONTRACT, USDC_CIRCLE_CONTRACT]

Ox_event_1_list_filtered = Ox_event_1_list.filter_tx_by_maker_taker_addresses(maker_taker_addresses, maker_taker_addresses)
Ox_event_2_list_filtered = Ox_event_2_list.filter_tx_by_maker_taker_addresses(maker_taker_addresses, maker_taker_addresses)
Ox_event_3_list_filtered = Ox_event_3_list.filter_tx_by_maker_taker_addresses(maker_taker_addresses, maker_taker_addresses)
Ox_event_4_list_filtered = Ox_event_4_list.filter_tx_by_maker_taker_addresses(maker_taker_addresses, maker_taker_addresses)


print(f"\nNumber of TransformedERC20 after filtered for USDC <> WETH: {Ox_event_1_list_filtered.list_length} (Correspond to {Ox_event_1_list_filtered.list_length/len(Ox_events_data_topic_1)*100:.1f}% of TransformedERC20 Events)")
print(f"Number of old RfqOrderFilled after filtered for USDC <> WETH: {Ox_event_2_list_filtered.list_length} (Correspond to {Ox_event_2_list_filtered.list_length/len(Ox_events_data_topic_2)*100:.1f}% of old RfqOrderFilled Events)")
print(f"Number of RfqOrderFilled after filtered for USDC <> WETH: {Ox_event_3_list_filtered.list_length} (Correspond to {Ox_event_3_list_filtered.list_length/len(Ox_events_data_topic_3)*100:.1f}% of RfqOrderFilled Events)")
print(f"Number of LimitOrderFilled after filtered for USDC <> WETH: {Ox_event_4_list_filtered.list_length} (Correspond to {Ox_event_4_list_filtered.list_length/len(Ox_events_data_topic_4)*100:.1f}% of LimitOrderFilled Events)")

SAVE_BIN = True
if SAVE_BIN is True:
    time_save0 = t0_sync.strftime("%y%m%d_at_%Hh%Mm%S")
    time_save1 = t1_sync.strftime("%y%m%d_at_%Hh%Mm%S")
    
    etherscanclient.save_data(Ox_event_1_list_filtered.make_pandas_for_calib(t0_sync, t1_sync), f"saved_data/0xProtocol_TransformedERC20Event_from{time_save0}_to{time_save1}")

    etherscanclient.save_data(Ox_event_2_list_filtered.make_pandas_for_calib(t0_sync, t1_sync), f"saved_data/0xProtocol_oldRfqOrderFilledEvent_from{time_save0}_to{time_save1}")

    etherscanclient.save_data(Ox_event_3_list_filtered.make_pandas_for_calib(t0_sync, t1_sync), f"saved_data/0xProtocol_RfqOrderFilledEvent_from{time_save0}_to{time_save1}")

    etherscanclient.save_data(Ox_event_4_list_filtered.make_pandas_for_calib(t0_sync, t1_sync), f"saved_data/0xProtocol_LimitOrderFilledEvent_from{time_save0}_to{time_save1}")