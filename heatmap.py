import asyncio
import multiprocessing
import os
import time
from collections import OrderedDict
from functools import partial
from multiprocessing import Pool
from threading import Lock

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

import numpy as np
from binance import Client, ThreadedDepthCacheManager, ThreadedWebsocketManager
from PIL import Image


def handle_dcm_message(depth_cache, liquidity_map):
    liquidity_map.dcm_callback_cnt += 1
    if liquidity_map.dcm_callback_cnt % liquidity_map.plot_dcm_every == 0:
        bids = depth_cache.get_bids()  # [[price, quantity], [price, quantity], ...]
        asks = depth_cache.get_asks()

        cur_price = (bids[0][0] + asks[0][0]) / 2
        price_bucket = liquidity_map.price_bucket
        rounded_cur_price = np.round(cur_price / price_bucket) * price_bucket
        liquidity_map.min_window_price = min(
            liquidity_map.min_window_price, rounded_cur_price
        )
        liquidity_map.max_window_price = max(
            liquidity_map.max_window_price, rounded_cur_price
        )

        bid_prices = np.array([bid[0] for bid in bids])
        bid_quantities = np.array([bid[1] for bid in bids])
        ask_prices = np.array([ask[0] for ask in asks])
        ask_quantities = np.array([ask[1] for ask in asks])

        all_prices = np.concatenate([bid_prices, ask_prices])
        all_quantities = np.concatenate([-bid_quantities, ask_quantities])
        rounded_prices = np.round(all_prices / price_bucket) * price_bucket

        liquidity_map.quantized_prices = np.unique(
            np.concatenate((liquidity_map.quantized_prices, rounded_prices))
        )
        quantized_quantities = np.zeros_like(liquidity_map.quantized_prices)
        for i, quantized_price in enumerate(liquidity_map.quantized_prices):
            quantized_quantities[i] = np.sum(
                all_quantities[rounded_prices == quantized_price]
            )

        liquidity_map.orderbook_lock.acquire()
        liquidity_map.quantized_depth_history.append(
            [
                liquidity_map.quantized_prices.copy(),
                quantized_quantities,
                rounded_cur_price,
            ]
        )
        liquidity_map.quantized_trades_history.append(
            {"trades": [], "quantized_sum": {}, "mm": {}}
        )
        liquidity_map.orderbook_lock.release()

        liquidity_map.min_quantized_price = min(
            liquidity_map.min_quantized_price, np.min(liquidity_map.quantized_prices)
        )
        liquidity_map.max_quantized_price = max(
            liquidity_map.max_quantized_price, np.max(liquidity_map.quantized_prices)
        )

        min_quantized_price_vis = max(
            liquidity_map.min_quantized_price,
            rounded_cur_price - liquidity_map.limit * liquidity_map.price_bucket,
        )
        max_quantized_price_vis = min(
            liquidity_map.max_quantized_price,
            rounded_cur_price + liquidity_map.limit * liquidity_map.price_bucket,
        )

        vis_h = (
            int((max_quantized_price_vis - min_quantized_price_vis) / price_bucket) + 2
        )
        full_h = (
            int(
                (liquidity_map.max_quantized_price - liquidity_map.min_quantized_price)
                / price_bucket
            )
            + 2
        )
        w = len(liquidity_map.quantized_depth_history)
        vis_heatmap = np.zeros((vis_h, w))
        for i, (prices, neg_quantities, rounded_price) in enumerate(
            liquidity_map.quantized_depth_history
        ):
            col = np.zeros(full_h)
            quantized_prices_idx = np.round(
                (prices - liquidity_map.min_quantized_price) / price_bucket
            ).astype(np.int32)
            col[quantized_prices_idx] = neg_quantities
            start = int(
                (min_quantized_price_vis - liquidity_map.min_quantized_price)
                / price_bucket
            )
            vis_heatmap[:, i] = col[start : start + vis_h]

        vis_heatmap = -vis_heatmap[::-1]  # flip price and quantity

        color_range = np.abs(vis_heatmap).max()
        plt.figure(figsize=(12, 8))
        plt.imshow(
            vis_heatmap,
            cmap="RdYlGn",
            interpolation="nearest",
            vmin=-color_range,
            vmax=color_range,
        )
        plt.colorbar()
        plt.title(
            f"Quantized Depth History for {liquidity_map.symbol}, prediction: {liquidity_map.prediction}"
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Price (USD)")
        plt.yticks(
            np.arange(0, vis_h, liquidity_map.price_label_gap),
            [f"{min_quantized_price_vis + i * price_bucket:.8f}" for i in range(vis_h)][
                :: -liquidity_map.price_label_gap
            ],
        )
        plt.xticks(
            np.arange(0, w, 10),
            [f"{i * liquidity_map.plot_dcm_every}" for i in np.arange(0, w, 10)],
        )
        for x, quantized_trade in enumerate(liquidity_map.quantized_trades_history):
            quantized_sum = quantized_trade["quantized_sum"]
            if len(quantized_sum) == 0:
                continue
            quantized_trade_prices = np.array(list(quantized_sum.keys()))
            quantized_trade_quantities = np.array(list(quantized_sum.values()))
            quantized_trade_time = [x] * len(quantized_trade_prices)
            plt.scatter(
                quantized_trade_time,
                (max_quantized_price_vis - quantized_trade_prices) / price_bucket,
                s=quantized_trade_quantities * liquidity_map.size_mult,
                c="blue",
                alpha=0.4,
            )
        plt.savefig(
            os.path.join(liquidity_map.save_path, f"{liquidity_map.symbol}.png")
        )
        plt.close()


def handle_socket_message(msg, liquidity_map):
    match msg["e"]:
        case "kline":
            pass
        case "trade":
            if len(liquidity_map.quantized_trades_history) > 0:
                liquidity_map.orderbook_lock.acquire()
                liquidity_map.quantized_trades_history[-1]["trades"].append(
                    [float(msg["p"]), float(msg["q"]), bool(msg["m"])]
                )
                liquidity_map.quantized_trades_history[-1]["quantized_sum"] = {}
                liquidity_map.quantized_trades_history[-1]["mm"] = {}
                quantized_sum = liquidity_map.quantized_trades_history[-1][
                    "quantized_sum"
                ]
                mm = liquidity_map.quantized_trades_history[-1]["mm"]
                for p, q, m in liquidity_map.quantized_trades_history[-1]["trades"]:
                    rounded_price = (
                        np.round(p / liquidity_map.price_bucket)
                        * liquidity_map.price_bucket
                    )
                    if rounded_price not in quantized_sum:
                        quantized_sum[rounded_price] = 0
                        mm[rounded_price] = 0
                    quantized_sum[rounded_price] += q
                    mm[rounded_price] += q if m else 0
                liquidity_map.orderbook_lock.release()
                liquidity_map.prediction = predict_market(liquidity_map)
        case "depth":
            pass
        case _:
            pass


def predict_market(liquidity_map, history_window=1, price_window=1, thres=2):
    for quantized_depth, quantized_trades in zip(
        liquidity_map.quantized_depth_history[-history_window:],
        liquidity_map.quantized_trades_history[-history_window:],
    ):
        quantity = -1
        cur_price = -1
        for p, q in quantized_trades[
            "quantized_sum"
        ].items():  # quantities are negative
            if q > quantity:
                quantity = q
                cur_price = p
        prices, quantities, _ = quantized_depth
        depth = dict(zip(prices, quantities))
        next_prices = (
            cur_price - np.arange(0, price_window, 1) * liquidity_map.price_bucket
        )
        vol = 0
        for p in next_prices:
            if p in depth:
                vol += depth[p]
        vol = -vol  # quantities are negated
        res = "none"
        if np.abs(vol) > thres:
            res = "up" if vol > 0 else "down"
        return res


class LiquidityMap:
    def __init__(
        self,
        symbol,
        client,
        plot_dcm_every=5,
        price_bucket=2,
        price_label_gap=5,
        limit=100,
        size_mult=5,
        save_path=".",
    ):
        self.symbol = symbol
        self.client = client
        self.liquidity_map = OrderedDict()

        self.plot_dcm_every = plot_dcm_every
        self.dcm_callback_cnt = 0
        self.price_bucket = price_bucket
        self.price_label_gap = price_label_gap
        self.limit = limit
        self.size_mult = size_mult
        self.save_path = save_path

        self.quantized_prices = np.zeros(0)
        self.min_quantized_price = 1e9
        self.max_quantized_price = -1e9
        self.min_window_price = 1e9
        self.max_window_price = -1e9
        self.orderbook_lock = Lock()
        self.quantized_depth_history = []
        self.quantized_trades_history = []
        self.prediction = None

        self.dcm = None
        self.twm = None

    def run(self, handle_dcm_message, handle_socket_message):
        self.dcm = ThreadedDepthCacheManager()
        self.dcm.start()

        twm_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(twm_loop)
        self.twm = ThreadedWebsocketManager(loop=twm_loop)
        self.twm.start()

        handle_dcm_message = partial(handle_dcm_message, liquidity_map=self)
        handle_socket_message = partial(handle_socket_message, liquidity_map=self)

        self.dcm.start_depth_cache(
            callback=handle_dcm_message,
            symbol=self.symbol,
            limit=self.limit,
            refresh_interval=1800,
        )
        self.twm.start_kline_socket(callback=handle_socket_message, symbol=self.symbol)
        self.twm.start_trade_socket(callback=handle_socket_message, symbol=self.symbol)
        self.twm.start_depth_socket(callback=handle_socket_message, symbol=self.symbol)

    def join(self):
        self.dcm.join()
        self.twm.join()


def launch_liquidity_map(args):
    api_key, api_secret, symbol, handle_dcm_message, handle_socket_message, lm_args = (
        args
    )
    client = Client(api_key, api_secret)
    lm = LiquidityMap(symbol, client, **lm_args)
    lm.run(
        handle_dcm_message=handle_dcm_message,
        handle_socket_message=handle_socket_message,
    )
    # lm.join()


if __name__ == "__main__":
    api_key = "haha"
    api_secret = "wow"
    configs = {
        "BTCUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 3,
            "price_label_gap": 5,
            "limit": 100,
            "size_mult": 5,
            "save_path": ".",
        },
        "ETHUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 0.2,
            "price_label_gap": 5,
            "limit": 100,
            "size_mult": 0.1,
            "save_path": ".",
        },
        "BNBUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 0.02,
            "price_label_gap": 10,
            "limit": 200,
            "size_mult": 0.2,
            "save_path": ".",
        },
        "SOLUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 0.01,
            "price_label_gap": 5,
            "limit": 100,
            "size_mult": 0.05,
            "save_path": ".",
        },
        "DOGEUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 1e-5,
            "price_label_gap": 20,
            "limit": 150,
            "size_mult": 3e-5,
            "save_path": ".",
        },
        "XRPUSDT": {
            "plot_dcm_every": 20,
            "price_bucket": 2e-4,
            "price_label_gap": 5,
            "limit": 100,
            "size_mult": 1e-4,
            "save_path": ".",
        },
    }
    worker_args = []
    for symbol, lm_args in configs.items():
        worker_args.append(
            (
                api_key,
                api_secret,
                symbol,
                handle_dcm_message,
                handle_socket_message,
                lm_args,
            )
        )
    with Pool(len(worker_args)) as p:
        p.map(launch_liquidity_map, worker_args)
        while True:
            time.sleep(4)
            imgs = []
            try:
                for symbol, arg in configs.items():
                    save_path = arg["save_path"]
                    img = Image.open(os.path.join(save_path, f"{symbol}.png"))
                    imgs.append(img)
                # Combine images
                height = 2
                width = len(imgs) // height
                total_width = width * imgs[0].width
                total_height = height * imgs[0].height
                new_im = Image.new("RGB", (total_width, total_height))
                x_offset = 0
                y_offset = 0
                for i, img in enumerate(imgs):
                    new_im.paste(img, (x_offset, y_offset))
                    x_offset += img.width
                    if (i + 1) % width == 0:
                        x_offset = 0
                        y_offset += img.height
                new_im.save(os.path.join(save_path, "combined.png"))
            except Exception as e:
                print(e)
                continue
        # p.join()
