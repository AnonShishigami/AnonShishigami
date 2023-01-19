#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 23:03:28 2022
Etherscan client to use API
@author: Shigami
"""

import pandas as pd
import pickle
import time
import warnings
import json
import requests
import numpy as np
from itertools import compress

from datetime import datetime
from datetime import timedelta
from web3 import Web3
from copy import deepcopy
# import sha3

from Crypto.Hash import keccak


warnings.filterwarnings(action='ignore', category=UserWarning)

def compute_event_hash(event_function_name):
    #k = sha3.keccak_256() # works with pysha3 library
    k = keccak.new(digest_bits=256)
    k.update(event_function_name)
    print(f"\nHash of {event_function_name} is '0x{k.hexdigest()}'")


def load_data(file_name):
    with open(file_name, "rb") as f:
        loaded_obj = pickle.load(f)

    return loaded_obj


def save_data(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

class EtherscanClient:
    """Client to query etherscan API."""

    BASE_URL = "http://api.etherscan.io/api"

    def __init__(self, api_key):
        self.api_key = api_key
        self.session = requests.Session()

    def build_url(self, module, action, **kwargs):
        """Generic function to make url."""
        url = f"{self.BASE_URL}?module={module}&action={action}&apikey={self.api_key}"

        for key, value in kwargs.items():
            url += f"&{key}={value}"

        return url

    def query_api(self, module, action, **kwargs):
        """Generic function to make a query to etherscan API."""
        url = self.build_url(module, action, **kwargs)
        url_response = self.session.get(url)
        if url_response.status_code == 200:
            data = url_response.json()

            if data["status"] == "1":
                return data["result"]
            else:
                if not data["result"]:
                    raise Exception(f"""Page empty: {data["message"]}""")
                else:
                    raise Exception(
                        f"""Invalid query: {data["result"]}""")
        else:
            raise Exception("Invalid url")

    def get_current_eth_price(self):
        """Query current ETH price computed by Etherscan."""
        data = self.query_api("stats", "ethprice")
        return data

    def _get_event_log_by_address(self, address, page, offset,
                                  fromBlock=0, toBlock="latests"):
        """
        Query event log by address.
        Returns the event logs from an address with optional filtering by
        block range.
        """
        kwargs = {"address": address,
                  "fromBlock": fromBlock,
                  "toBlock": toBlock,
                  "page": page,
                  "offset": offset}
        event_logs = self.query_api("logs", "getLogs", **kwargs)

        return event_logs

    def get_event_log_by_address(self, address, page, offset,
                                 fromBlock=0, toBlock="latests"):
        """Get event logs and convert them to easily readable format."""
        events_logs = self._get_event_log_by_address(
            address, page, offset, toBlock="latests")
        events = []
        for event in events_logs:
            events.append(EventLogsEtherscan(event))

        return events

    def _get_event_log_by_address_filter_topics(self, address, page, offset,
                                                topic0, topic1="", topic2="",
                                                fromBlock=0, toBlock="latests"):
        """
        Query event log by address and filtered by topics.
        Rturns the event logs from an address, filtered by topics and block
        range.
        """

        kwargs = {"address": address,
                  "fromBlock": fromBlock,
                  "toBlock": toBlock,
                  "page": page,
                  "offset": offset,
                  "topic0": topic0}
        # TODO make topics dict which adapt to the number of topics
        event_logs_topics = self.query_api("logs", "getLogs", **kwargs)

        return event_logs_topics

    def get_event_log_by_address_filtered_topics(self, address, topic0,
                                                 offset=1000, topic1="", topic2="",
                                                 fromBlock=0, toBlock="latests"):
        """Get event logs filtered by topics and convert them to easily readable format."""
        page = 1
        events_logs_topics = self._get_event_log_by_address_filter_topics(
            address, page, offset, topic0, topic1="", topic2="",
            fromBlock=fromBlock, toBlock=toBlock)

        length_page = len(events_logs_topics)
        while length_page == offset:
            print(f"Page {page} is full. Querying page {page+1}")
            page += 1

            events_logs_topics_new_page = self._get_event_log_by_address_filter_topics(
                address, page, offset, topic0, topic1="", topic2="",
                fromBlock=fromBlock, toBlock=toBlock)

            events_logs_topics.extend(events_logs_topics_new_page[:])

            length_page = len(events_logs_topics_new_page)

        events_topic = []
        for event in events_logs_topics:
            events_topic.append(EventLogsEtherscan(event))

        return events_topic

    def get_contract_ABI(self, address):
        kwargs = {"address": address}
        ABI = self.query_api("contract", "getabi", **kwargs)
        return ABI

    def get_block_number_by_timestamp(self, timestamp):
        unix_timestamp = int(time.mktime(timestamp.timetuple()))
        kwargs = {"timestamp": unix_timestamp,
                  "closest": "before"}
        return int(self.query_api("block", "getblocknobytime", **kwargs))

    def convert_hex_to_int(self, number_hex):
        return int(number_hex, 16)

    def convert_timestamp_to_date(self, timestamp):
        return datetime.fromtimestamp(int(timestamp))

    def get_block_info_timestamp(self, block_number):
        kwargs = {"boolean": "true",
                "tag": hex(block_number)}
        url = self.build_url(module="proxy", action="eth_getBlockByNumber", **kwargs)
        url_response = self.session.get(url)
        
        if url_response.status_code == 200:
            data = url_response.json()
        else:
            raise Exception("Invalid url")

        return data["result"]["timestamp"]


class EventLogsEtherscan:
    """Class for transforming event logs to human redable data."""

    def __init__(self, event):
        self.address = event["address"]
        self.block_hash = event["blockHash"]
        self.block_number = int(event["blockNumber"], 16)
        self.data = event["data"]
        self.gas_price = int(event["gasPrice"], 16)
        self.gas_price_Gwei = int(event["gasPrice"], 16)/10**9
        self.gas_used = int(event["gasUsed"], 16)
        self.log_index = int(event["logIndex"], 16)
        self.timestamp = int(event["timeStamp"], 16)
        self.timestamp_date = datetime.fromtimestamp(int(self.timestamp))
        self.topics = event["topics"]
        self.tx_hash = event["transactionHash"]

        # issue here somtimes, maybe bc of page issue ? need to be tested
        # but this attributdoes not matter anyway

        # try:
        #     self.tx_index = int(event["transactionIndex"], 16)
        # except:
        #     print(event["transactionIndex"])
        #     raise Exception(ValueError)

    def decode_event_logs(self, abi, w3_session, event_name):
        """
        Decode data field of an event log for a given smart contract.
        Return event log with decoded data field.
        """
        # Get transaction receipt
        receipt = w3_session.eth.get_transaction_receipt(self.tx_hash)

        # Get corresponding event log in the transaction
        logs = receipt["logs"]  # [f"self.log_index"]
        for index_log, log_search in enumerate(logs):
            if int(log_search["logIndex"]) == self.log_index:
                log = receipt["logs"][index_log]

        # Create contract object
        smart_contract = log["address"]
        contract = w3_session.eth.contract(smart_contract, abi=abi)

        abi_events = [abi["name"]
                      for abi in contract.abi if abi["type"] == "event"]

        if event_name in abi_events:
            decoded_fields = contract.events[event_name](
            ).processReceipt(receipt)
        else:
            raise(
                Exception(f"Given event name does not exist in ABI : {event_name}"))

        # Select only decoded event log corresponding to the input event log
        for index_decoded, decoded_field_search in enumerate(decoded_fields):
            if int(decoded_field_search["logIndex"]) == self.log_index:
                decoded_field_log = decoded_fields[index_decoded]

        decoded_event_log = deepcopy(self)
        decoded_event_log.data = vars(decoded_field_log["args"])

        return decoded_event_log

    def decode_event_logs_manual_parser(self, topics0_hash):
        """
        Decode data field of an event log for a given smart contract with a manually edited parser.
        Return event log with decoded data field.
        """
        # def parse_log(data, topics0):
        data = self.data
        nb_slot = int((len(data)-2)/64)
        slots = [data[2+i*64:2+(1+i)*64] for i in range(nb_slot)]

        topics = {"0xTransformedERC20Event" : "0x0f6672f78a59ba8e5e5b5d38df3ebc67f3c792e2c9259b8d97d7f00dd78ba1b3",
                    "0x event 2" : "0xac75f773e3a92f1a02b12134d65e1f47f8a14eabe4eaf1e24624918e6a8b269f",
                    "0xRfqOrderFilledEvent" : "0x829fa99d94dc4636925b38632e625736a614c154d55006b7ab6bea979c210c32",
                    "0xLimitOrderFilledEvent" : "0xab614d2b738543c0ea21f56347cf696a3a0c42a7cbec3212a5ca22a4dcff2124"}

        if topics0_hash == topics["0xTransformedERC20Event"]:
            swap_event = {
                "makerToken": '0x' + slots[0][24:], # token_in
                "takerToken": '0x' + slots[1][24:], # token_out
                "takerTokenFilledAmount": int(slots[2], 16), # amount_in
                "makerTokenFilledAmount": int(slots[3], 16), # amount_out
                "tx hash" : self.tx_hash
            }
            return swap_event

        elif topics0_hash == topics["0x event 2"] or topics0_hash == topics["0xRfqOrderFilledEvent"]:
            RfqOrderFilled_event = {
                "0x orderHash": "0x" + slots[0][24:],
                "maker": "0x" + slots[1][24:],
                "taker": "0x" + slots[2][24:],
                "makerToken": "0x" + slots[3][24:],
                "takerToken": "0x" + slots[4][24:],
                "takerTokenFilledAmount": int(slots[5], 16),
                "makerTokenFilledAmount": int(slots[6], 16),
                "tx hash" : self.tx_hash
            }
            if topics0_hash == topics["0xRfqOrderFilledEvent"]:
                RfqOrderFilled_event["pool"] = int(slots[7], 16)

            return RfqOrderFilled_event

        elif topics0_hash == topics["0xLimitOrderFilledEvent"]:
            LimitOrderFilled_event = {
                "orderHash": "0x" + slots[0][24:],
                "maker": "0x" + slots[1][24:],
                "taker": "0x" + slots[2][24:],
                "feeRecipient": "0x" + slots[3][24:],
                "makerToken": "0x" + slots[4][24:],
                "takerToken": "0x" + slots[5][24:],
                "takerTokenFilledAmount": int(slots[6], 16),
                "makerTokenFilledAmount": int(slots[7], 16),
                "takerTokenFeeFilledAmount": int(slots[8], 16),
                "protocolFeePaid": int(slots[9], 16),
                "pool": int(slots[10], 16),
                "tx hash" : self.tx_hash
            }
            return LimitOrderFilled_event

        else:
            raise ValueError(f"unrecognized topic0: {topics0_hash}")


    def get_contract_abi(self, contract_address, api_key_etherscan):
        abi_endpoint = f"https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={api_key_etherscan}"
        abi = json.loads(requests.get(abi_endpoint).text)

        if abi["status"] == "1":
            return abi["result"]
        elif abi["status"] == "0":
            raise Exception(f"""{abi["message"]}""")
        else:
            raise Exception(
                f"""Error in abi status : {abi["status"]}. Get following error message {abi["message"]}""")




class UniSyncEvent(EventLogsEtherscan):
    """Class representing a Sync event on Uniswap."""

    def __init__(self, event):
        self.address = event.address
        self.block_hash = event.block_hash
        self.block_number = event.block_number
        self.data = event.data
        self.gas_price = event.gas_price
        self.gas_price_Gwei = event.gas_price
        self.gas_used = event.gas_used
        self.log_index = event.log_index
        self.timestamp = event.timestamp
        self.timestamp_date = event.timestamp_date
        self.topics = event.topics
        self.tx_hash = event.tx_hash

    def get_reserves(self, decimals0=10**6, decimals1=10**18):
        reserve0 = self.data["reserve0"]/decimals0
        reserve1 = self.data["reserve1"]/decimals1
        return reserve0, reserve1

    def compute_ETH_price(self):
        reserve0, reserve1 = self.get_reserves()
        return reserve0/reserve1

    def compute_liquidity(self):
        if self.address == "0xb4e16d0168e52d35cacd2c6185b44281ec28c9dc":

            return 2*self.reserve0

        else:
            raise(
                Exception(f"Only ETH/USDC uniV2 implemented yet. Address: {self.address}"))


class UniSwappingEvent(EventLogsEtherscan):
    """Class representing a Swap event on Uniswap."""

    def __init__(self, event):
        self.address = event.address
        self.block_hash = event.block_hash
        self.block_number = event.block_number
        self.data = event.data
        self.gas_price = event.gas_price
        self.gas_price_Gwei = event.gas_price
        self.gas_used = event.gas_used
        self.log_index = event.log_index
        self.timestamp = event.timestamp
        self.timestamp_date = event.timestamp_date
        self.topics = event.topics
        self.tx_hash = event.tx_hash

    def get_amounts(self, decimals0=10**6, decimals1=10**18):
        amount0_in, amount1_in = self._get_amounts_in()
        amount0_out, amount1_out = self._get_amounts_out()
        if amount0_in != 0:
            print(f"Swapped {amount0_in:.2f} USDC for {amount1_out:5f} ETH")
        elif amount1_in != 0:
            print(f"Swapped {amount1_in:5f} ETH for {amount0_out:.2f} USDC")
        else:
            raise(Exception("No amount in"))

        return amount0_in, amount1_in, amount0_out, amount1_out

    def _get_amounts(self, decimals0=10**6, decimals1=10**18):
        amount0_in, amount1_in = self._get_amounts_in()
        amount0_out, amount1_out = self._get_amounts_out()

        return amount0_in, amount1_in, amount0_out, amount1_out

    def _get_amounts_in(self, decimals0=10**6, decimals1=10**18):
        amount0_in = self.data["amount0In"]/decimals0
        amount1_in = self.data["amount1In"]/decimals1

        return amount0_in, amount1_in

    def _get_amounts_out(self, decimals0=10**6, decimals1=10**18):
        amount0_out = self.data["amount0Out"]/decimals0
        amount1_out = self.data["amount1Out"]/decimals1

        return amount0_out, amount1_out

    def compute_event_volume_USDC(self, decimals0=10**6, decimals1=10**18):
        """
        Compute volume of a transaction in USDC.
        Return amount of USDC exchanged. Positive sign for ETH bought
        and negative sign for ETH sold.
        """

        amount0_in, amount1_in, amount0_out, amount1_out = self._get_amounts()

        if amount0_in != 0:
            return amount0_in
        elif amount1_in != 0:
            return -amount0_out
        else:
            raise(Exception("No amount in"))
        # TODO check if it is correct because of the fees. Probably need
        # to change something to combien infos from Swap and Sync


class OxProtocolEvent(EventLogsEtherscan):
    """
    Class representing the events from 0x protocol.
    """
    def __init__(self, event):
        self.address = event.address
        self.block_hash = event.block_hash
        self.block_number = event.block_number
        self.data = event.data
        self.gas_price = event.gas_price
        self.gas_price_Gwei = event.gas_price
        self.gas_used = event.gas_used
        self.log_index = event.log_index
        self.timestamp = event.timestamp
        self.timestamp_date = event.timestamp_date
        self.topics = event.topics
        self.tx_hash = event.tx_hash
        self.decoded_data = event.decode_event_logs_manual_parser(event.topics[0])
        self.WETH_MAKER_CONTRACT_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2".lower()
        self.USDC_CIRCLE_CONTRACT_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48".lower()


    def compute_price(self):
        if self.topics[0].lower() == "0x829fa99d94dc4636925b38632e625736a614c154d55006b7ab6bea979c210c32": 
            if self.decoded_data["makerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
                return self.decoded_data["makerTokenFilledAmount"]*10**-18/(self.decoded_data["takerTokenFilledAmount"]*10**-6)
            elif self.decoded_data["makerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
                return self.decoded_data["makerTokenFilledAmount"]*10**-6/(self.decoded_data["takerTokenFilledAmount"]*10**-18)
            else:
                return self.decoded_data["makerTokenFilledAmount"]/self.decoded_data["takerTokenFilledAmount"]

        elif self.topics[0].lower() == "0x0f6672f78a59ba8e5e5b5d38df3ebc67f3c792e2c9259b8d97d7f00dd78ba1b3" or "0xac75f773e3a92f1a02b12134d65e1f47f8a14eabe4eaf1e24624918e6a8b269f":
            if self.decoded_data["makerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
                return self.decoded_data["makerTokenFilledAmount"]*10**-6/(self.decoded_data["takerTokenFilledAmount"]*10**-18)
            elif self.decoded_data["makerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
                return self.decoded_data["makerTokenFilledAmount"]*10**-18/(self.decoded_data["takerTokenFilledAmount"]*10**-6)
            else:
                return self.decoded_data["makerTokenFilledAmount"]/self.decoded_data["takerTokenFilledAmount"]

    # def get_convert_Token_Filled_Amount(self, ):
    #     if self.decoded_data["makerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
    #         return {"WETHMakerTokenFilledAmount": self.decoded_data["makerTokenFilledAmount"]*10**-18,
    #         "USDCTakerTokenFilledAmount": self.decoded_data["takerTokenFilledAmount"]*10**-6}
    #     elif self.decoded_data["makerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS and self.decoded_data["takerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
    #         return {"USDCMakerTokenFilledAmount": self.decoded_data["makerTokenFilledAmount"]*10**-6,
    #         "WETHTakerTokenFilledAmount": self.decoded_data["takerTokenFilledAmount"]*10**-18}

    def convert_maker_filled_amount(self):
        if self.decoded_data["makerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
            return self.decoded_data["makerTokenFilledAmount"]*10**-18
        elif self.decoded_data["makerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
            return self.decoded_data["makerTokenFilledAmount"]*10**-6
        else:
            raise Exception("Unknown maker token address")

    def convert_maker_filled_currency(self):
        if self.decoded_data["makerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
            return "WETH"
        elif self.decoded_data["makerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
            return "USDC"
        else:
            raise Exception("Unknown maker token address")

    def convert_taker_filled_amount(self):
        if self.decoded_data["takerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
            return self.decoded_data["takerTokenFilledAmount"]*10**-18
        elif self.decoded_data["takerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
            return self.decoded_data["takerTokenFilledAmount"]*10**-6
        else:
            raise Exception("Unknown taker token address")

    def convert_taker_filled_currency(self):
        if self.decoded_data["takerToken"].lower() == self.WETH_MAKER_CONTRACT_ADDRESS:
            return "WETH"
        elif self.decoded_data["takerToken"].lower() == self.USDC_CIRCLE_CONTRACT_ADDRESS:
            return "USDC"
        else:
            raise Exception("Unknown taker token address")


class SwapEventList:
    """Represents a list of SwapEvents and perform bulk operations on the list"""

    def __init__(self, swap_event_list):
        self.swap_event_list = swap_event_list

    def compute_histogram_volume(self, time_bin_size):
        """
        Compute values for histogram of effective volume with a given binning.
        Returns total, net volume and bin edges to plot histograms.
        """
        if type(time_bin_size) != type(timedelta()):
            raise(Exception(
                f"Invalid type of time_bin_size. Must be a datetime.timedelta and type {type(time_bin_size)} was given."))

        start_bin = self.swap_event_list[0].timestamp_date
        end_bin = self.swap_event_list[-1].timestamp_date
        number_of_bins = round((end_bin-start_bin)/time_bin_size)
        bins_limits = [start_bin+i *
                       time_bin_size for i in range(number_of_bins+1)]

        total_volume = np.zeros(number_of_bins)
        net_volume = np.zeros(number_of_bins)
        index_while = 0

        # Compute volume according to bins edges defined above.
        for index_for, timestamp_bin_limit in enumerate(bins_limits[:-1]):
            while self.swap_event_list[index_while].timestamp_date < timestamp_bin_limit:
                total_volume[index_for] += abs(
                    self.swap_event_list[index_while].compute_event_volume_USDC())
                net_volume[index_for] += self.swap_event_list[index_while].compute_event_volume_USDC()

                index_while += 1

        return total_volume, net_volume, bins_limits

    def get_event_large_transactions(self, treshold):
        """Select swap events in the list above the threshold value."""

        selectionned = [event.compute_event_volume_USDC() >
                        treshold for event in self.swap_event_list]

        large_transaction_event_list = [large_transaction for large_transaction, indicator in zip(
            self.swap_event_list, selectionned) if indicator == True]
        return SwapEventList(large_transaction_event_list)


class SyncEventList:
    """Represents a list of SyncEvents and perform bulk operations on the list"""

    def __init__(self, sync_event_list):
        self.sync_event_list = sync_event_list

    def get_exchange_ratio(self):
        exchange_ratio = [event.compute_ETH_price()
                          for event in self.sync_event_list]

        return exchange_ratio

    def get_timestamps(self):
        return [event.timestamp for event in self.sync_event_list]

    def get_pair_over_timestamp_data(self):
        return self.get_timestamps(), self.get_exchange_ratio()

    def get_reserves_list(self):
        reserve0 = []
        reserve1 = []
        for event in self.sync_event_list:
            reserve0_loop, reserve1_loop = event.get_reserves()
            reserve0.append(reserve0_loop)
            reserve1.append(reserve1_loop)

        return reserve0, reserve1

    def get_pair_over_date_data(self):
        return self.get_timestamp_date(), self.get_exchange_ratio()

    def get_timestamp_date(self):
        return [event.timestamp_date for event in self.sync_event_list]



class OxProtocolClientList():
    def __init__(self, Ox_protocol_event_list):
        self.Ox_protocol_event_list = Ox_protocol_event_list
        self.list_length = len(self.Ox_protocol_event_list)

    def get_exchange_ratio_list(self):
        return [self.Ox_protocol_event_list[i].compute_price() for i in range(self.list_length)]

    def get_list_tx_hash(self):
        return [self.Ox_protocol_event_list[i].tx_hash for i in range(self.list_length)]

    def is_maker(self, maker_address_list):
        maker_address_list_lower = [address.lower() for address in maker_address_list]
        return[self.Ox_protocol_event_list[i].decoded_data["makerToken"].lower() in maker_address_list_lower for i in range(self.list_length)]

    def is_taker(self, taker_address_list):
        taker_address_list_lower = [address.lower() for address in taker_address_list]
        return[self.Ox_protocol_event_list[i].decoded_data["takerToken"].lower() in taker_address_list_lower for i in range(self.list_length)]

    def is_taker_or_maker(self, maker_address_list, taker_address_list):
        is_maker = self.is_maker(maker_address_list)
        is_taker = self.is_taker(taker_address_list)
        return [is_maker[i] & is_taker[i] for i in range(self.list_length)]

    def filter_tx_by_maker_taker_addresses(self, maker_address_list, taker_address_list):
        return OxProtocolClientList(list(compress(self.Ox_protocol_event_list, self.is_taker_or_maker(maker_address_list, taker_address_list))))
         
    def make_pandas_for_calib(self, t0_acq, t1acq):
        clean_data = {
            "BlockNumber": [self.Ox_protocol_event_list[i].block_number for i in range(self.list_length)],
            "Tx_hash": self.get_list_tx_hash(),
            "Computed_price": self.get_exchange_ratio_list(),
            "makerTokenFilledAmount": [self.Ox_protocol_event_list[i].decoded_data["makerTokenFilledAmount"] for i in range(self.list_length)],
            "takerTokenFilledAmount": [self.Ox_protocol_event_list[i].decoded_data["takerTokenFilledAmount"] for i in range(self.list_length)],
            "convertedMakerTokenFilledAmount": self.get_convert_maker_filled_amount_list(),
            "convertedTakerTokenFilledAmount": self.get_convert_taker_filled_amount_list(),
            "makerToken": [self.Ox_protocol_event_list[i].decoded_data["makerToken"] for i in range(self.list_length)],
            "convertedMakerTokenFilledCurrency": self.get_convert_maker_filled_currency_list(),
            "convertedTakerTokenFilledCurrency": self.get_convert_taker_filled_currency_list(),
            "makerToken": [self.Ox_protocol_event_list[i].decoded_data["makerToken"] for i in range(self.list_length)],
            "takerToken": [self.Ox_protocol_event_list[i].decoded_data["takerToken"] for i in range(self.list_length)],
            "Topics0": [self.Ox_protocol_event_list[i].topics[0] for i in range(self.list_length)],
            "Contract_address": [self.Ox_protocol_event_list[i].address for i in range(self.list_length)],
            "TX_timestamps": [self.Ox_protocol_event_list[i].timestamp for i in range(self.list_length)],
            "TX_timestamps_date": [self.Ox_protocol_event_list[i].timestamp_date for i in range(self.list_length)],
            "Query start date": t0_acq,
            "Query end date": t1acq
        }
        clean_data_sorted = pd.DataFrame(clean_data).sort_values(by=['BlockNumber'], ignore_index=True)
        return pd.DataFrame(clean_data_sorted)

    def get_convert_maker_filled_amount_list(self):
        return [self.Ox_protocol_event_list[i].convert_maker_filled_amount() for i in range(self.list_length)]

    def get_convert_maker_filled_currency_list(self):
        return [self.Ox_protocol_event_list[i].convert_maker_filled_currency() for i in range(self.list_length)]

    def get_convert_taker_filled_amount_list(self):
        return [self.Ox_protocol_event_list[i].convert_taker_filled_amount() for i in range(self.list_length)]
 
    def get_convert_taker_filled_currency_list(self):
        return [self.Ox_protocol_event_list[i].convert_taker_filled_currency() for i in range(self.list_length)]



class ChainLinkClient:
    """Client to interact easily with Chainlink contracts"""

    def __init__(self, provider_url, address, abi):
        self.address = address
        self.abi = abi
        self.w3_session = Web3(Web3.HTTPProvider(provider_url))
        self.contract = self._connect_to_contract()

    def _connect_to_contract(self):
        contract = self.w3_session.eth.contract(
            address=self.address, abi=self.abi)

        return contract

    def get_decimals(self):
        return self.contract.functions.decimals().call()

    def get_round_data(self, roundID):
        return self.contract.functions.getRoundData(roundID).call()

    def get_latest_round_data(self):
        return self.contract.functions.latestRoundData().call()

    def get_pair_description(self):
        return self.contract.functions.description().call()

    def get_latestTransmissionDetails(self):
        return self.contract.functions.latestTransmissionDetails().call()

    def get_transmitters(self):
        return self.contract.functions.transmitters().call()


