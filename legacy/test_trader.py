from indicators import *
import numpy as np
from numba import njit
import h5py
from statistics import mean
from config import FILES_DIR
import finplot as fplt
import random
from datetime import timezone
import pandas as pd


@njit
def pip_calc(start_price, end_price, multiplier, spread=0):
    return (end_price - start_price) / multiplier - spread


class Currency:

    def __init__(self, pair, strategy, verbose=False, cushion=2, **kw):
        self.name = pair
        self.verbose = verbose
        self.params = kw["params"]
        self.strategy_dict = strategy
        # self.start_pos = round(max(self.params[2:8]))
        self.start_pos = 0

        with h5py.File(f"{FILES_DIR}pairs_data.h5", 'r') as f:
            self.df = {tf: np.array(f[f"{tf}_{pair}"]) for tf in self.strategy_dict["tfs"]}

        self.config = self.account_data = pd.read_csv(FILES_DIR + "account_data.csv", index_col=0)

        # Main
        self.index = index_to_datetime(self.df[self.strategy_dict["tfs"][0]]["time"])
        self.open = self.df[self.strategy_dict["tfs"][0]]['open']
        self.high = self.df[self.strategy_dict["tfs"][0]]['high']
        self.low = self.df[self.strategy_dict["tfs"][0]]['low']
        self.close = self.df[self.strategy_dict["tfs"][0]]['close']
        self.vol = self.df[self.strategy_dict["tfs"][0]]['vol']
        self.spread = self.df[self.strategy_dict["tfs"][0]]['spread'] / 10

        # # M5
        # self.m5_index = self.df["M5"]["data"]["time"]
        # self.m5_open = self.df["M5"]["data"]['open']
        # self.m5_high = self.df["M5"]["data"]['high']
        # self.m5_low = self.df["M5"]["data"]['low']
        # self.m5_close = self.df["M5"]["data"]['close']
        # self.m5_vol = self.df["M5"]["data"]['vol']

        # # M15
        # self.m15_index = self.df["M15"]["time"]
        # self.m15_open = self.df["M15"]['open']
        # self.m15_high = self.df["M15"]['high']
        # self.m15_low = self.df["M15"]['low']
        # self.m15_close = self.df["M15"]['close']
        # self.m15_vol = self.df["M15"]['vol']

        # # M30
        # self.m30_index = self.df["M30"]["data"]["time"]
        # self.m30_open = self.df["M30"]["data"]['open']
        # self.m30_high = self.df["M30"]["data"]['high']
        # self.m30_low = self.df["M30"]["data"]['low']
        # self.m30_close = self.df["M30"]["data"]['close']
        # self.m30_vol = self.df["M30"]["data"]['vol']

        # # H1
        # self.h1_index = self.df["H1"]["time"]
        # self.h1_open = self.df["H1"]['open']
        # self.h1_high = self.df["H1"]['high']
        # self.h1_low = self.df["H1"]['low']
        # self.h1_close = self.df["H1"]['close']
        # self.h1_vol = self.df["H1"]['vol']

        # # H4
        # self.h4_index = self.df["H4"]["data"]["time"]
        # self.h4_open = self.df["H4"]["data"]['open']
        # self.h4_high = self.df["H4"]["data"]['high']
        # self.h4_low = self.df["H4"]["data"]['low']
        # self.h4_close = self.df["H4"]["data"]['close']
        # self.h4_vol = self.df["H4"]["data"]['vol']

        # # D1
        # self.d1_index = self.df["D1"]["time"]
        # self.d1_open = self.df["D1"]['open']
        # self.d1_high = self.df["D1"]['high']
        # self.d1_low = self.df["D1"]['low']
        # self.d1_close = self.df["D1"]['close']
        # self.d1_vol = self.df["D1"]['vol']

        if str(self.open[-1]).index(".") >= 3:  # JPY pair
            self.multiplier = 0.01
        else:
            self.multiplier = 0.0001

        # self.open_ha, self.high_ha, self.low_ha, self.close_ha = heikin_ashi(
        #     self.open, self.high, self.low, self.close
        # )

        # Hunter
        # self.tp = round(self.params[0])
        # self.sl = round(self.params[1])
        # self.tp = 200
        # self.sl = 20
        # self.atr_period = round(self.params[2])
        # self.tma_slope_period = round(self.params[3])
        # self.m1_tma_slope_level_long = round(self.params[4], 6)
        # self.m1_tma_slope_level_short = round(self.params[5], 6)
        # self.m1_tma_core_entry_slope = round(self.params[6], 6)
        # self.m1_tma_pattern_entry_slope = round(self.params[7], 6)
        # self.m1_tma_width_pattern_entry = round(self.params[8], 6)
        # self.m1_tma_width_core_entry = round(self.params[9], 6)
        # self.m1_tma_width_entry_level = round(self.params[10], 6)
        # self.strength_period = round(self.params[11])
        # self.strength_level_long = round(self.params[12], 3)
        # self.strength_level_short = round(self.params[13], 3)
        # self.m5_tma_slope_level = round(self.params[14], 6)
        # self.atr_multiplier = round(self.params[15])
        # self.use_exhaustion = round(self.params[16])
        # self.max_previous_bar_size = round(self.params[17])
        # self.start_hour = round(self.params[18])
        # self.end_hour = round(self.params[19])
        # self.m1_tma_slope_long_slow_stop = round(self.params[20], 6)
        # self.m1_tma_slope_long_medium_stop = round(self.params[21], 6)
        # self.m1_tma_slope_long_fast_stop = round(self.params[22], 6)
        # self.m1_tma_slope_short_slow_stop = round(self.params[23], 6)
        # self.m1_tma_slope_short_medium_stop = round(self.params[24], 6)
        # self.m1_tma_slope_short_fast_stop = round(self.params[25], 6)

        self.tp = 200
        self.sl = 30
        self.tp_multiplier = 1.5
        self.slope_level = 0.0002
        # self.tema_period = round(self.params[2])
        # self.ema_period = round(self.params[3])
        # self.fast_ema_period = round(self.params[4])
        # self.slow_ema_period = round(self.params[5])
        # self.signal_ema_period = round(self.params[6])
        # self.atr_period = 100
        # self.atr_multiplier = 7

        # self.trailing_stop = 20 * self.multiplier
        self.break_even = 25
        self.atr_multiplier = 5  # 1-30
        self.atr_period = 21  # 10-100
        # self.tma_slope_period = 1  # 1-10
        # self.m1_tma_slope_entry = 0.00005  # 0-1
        # self.m1_tma_slope_exit = 0.00001  # 0-1
        # self.m1_tma_width_entry = 0.0009  # 0-1
        # self.m1_tma_width_exit = 0.0001  # 0-1
        # self.strength_period = 21
        # self.strength_level_long = 0
        # self.strength_level_short = 0
        # self.m5_tma_slope_level = 0
        # self.use_exhaustion = 1
        # self.track_exits = 0
        # self.max_previous_bar_size = 10
        # self.start_hour = 0
        # self.end_hour = 25

        self.trailing_stop = 30
        # self.tp = self.params[0]
        # self.sl = self.params[1]
        # self.trailing_stop = self.params[2]
        # self.atr_period = self.params[3]
        # self.rsi_period = self.params[4]
        # self.rsi_period = 14
        # self.mfi_period = self.params[5]
        # self.small_ema_period = self.params[6]
        # self.big_ema_period = self.params[7]
        # self.wr_period = self.params[8]
        # self.long_rsi_level = self.params[9]
        # self.short_rsi_level = self.params[10]
        # self.long_mfi_level = self.params[11]
        # self.short_mfi_level = self.params[12]
        # self.long_wr_level = self.params[13]
        # self.short_wr_level = self.params[14]
        # self.atr_multiplier = self.params[17]

        # self.tfs_list = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        # self.tfs_list = ["D1"]  # ["H1", "M15", "M1"]
        self.tfs_list = ["M1", "M15", "H1", "H4"]

        self.colors = {
            "M1": "#F7ECDE", "M5": "yellow", "M15": "red", "M30": "purple", "H1": "#ADD8E6", "H4": "brown",
            "D1": "white"
        }

        self.indicators_dict = {
            "hma": hma, "tma": tma, "tsv": tsv, "ichimoku_cloud": ichimoku_cloud, "wr": get_wr,
            "supertrend": supertrend, "tma_slope_middle": slope, "hma_slope_middle": slope,
            "mml": mml_calculator,
        }
        self.tfs_dict = {
            # "hma": self.tfs_list,
            "tma": self.tfs_list,
            # "tsv": self.tfs_list,
            "ichimoku_cloud": self.tfs_list,
            # "mml": self.tfs_list,
            # "wr": self.tfs_list,
            # "supertrend": self.tfs_list,
            # "tma_slope_middle": self.tfs_list,
            # "hma_slope_middle": self.tfs_list,
        }

        for indicator, tf_list in self.tfs_dict.items():
            for tf in tf_list:
                self.indicators_input = {
                    "hma": {"c": self.df[tf]["close"], "n": 55},
                    "tma": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"]},
                    "tsv": {"c": self.df[tf]["close"], "v": self.df[tf]["vol"]},
                    "ichimoku_cloud": {"h": self.df[tf]["high"], "l": self.df[tf]["low"]},
                    "wr": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"], "n": 105},
                    "supertrend": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"]},
                    "mml": {"h": self.df[tf]["high"], "l": self.df[tf]["low"]},
                }

                # if hasattr(self, f"tma_{tf.lower()}"):
                #     self.indicators_input["tma_slope_middle"] = {"value": getattr(self, f"tma_{tf.lower()}")[1], "n": 1}
                #     self.indicators_input["hma_slope_middle"] = {"value": getattr(self, f"hma_{tf.lower()}")[1], "n": 1}

                setattr(self, f"{indicator}_{tf.lower()}",
                        self.indicators_dict[indicator](**self.indicators_input[indicator]))

        # M1
        # setattr(self, "hma_m1", hma(self.df["M1"]["close"], 55))
        # setattr(self, "hma_slope_m1", slope(self.hma_m1, 1))

        # for tf in self.strategy_dict["tfs"]:
        #     for indicator in self.strategy_dict["indicators"][tf]:
        #         setattr(self, f"{indicator}_{tf.lower()}", hma(self.df[tf]["close"], 55))
        #         setattr(self, "hma_slope_" + tf.lower(), slope(getattr(self, "hma_" + tf.lower()), 1))

        self.indicators = ["hma_slope", "tma_slope", "tsv", "ichimoku", "wr_21", "supertrend"]

        # for tf in self.tfs_list:
        #     setattr(self, f"hma_{tf.lower()}", hma(self.df[tf]["close"], 55))
        #     setattr(self, f"hma_slope_{tf.lower()}", slope(getattr(self, f"hma_{tf.lower()}"), 1))
        #
        #     setattr(self, f"tma_{tf.lower()}", tma(self.df[tf]["high"], self.df[tf]["low"], self.df[tf]["close"]))
        #     setattr(self, f"tma_slope_{tf.lower()}", slope(getattr(self, f"tma_{tf.lower()}")[1], 1))
        #
        #     setattr(self, f"tsv_{tf.lower()}", tsv(self.df[tf]["close"], self.df[tf]["vol"])[0])
        #
        #     setattr(self, f"ichimoku_{tf.lower()}", ichimoku_cloud(self.df[tf]["high"], self.df[tf]["low"]))
        #
        #     setattr(self, f"wr_21_{tf.lower()}", get_wr(self.df[tf]["high"], self.df[tf]["low"], self.df[tf]["close"]))
        #
        #     setattr(self, f"supertrend_{tf.lower()}", supertrend(self.df[tf]["high"], self.df[tf]["low"],
        #                                                          self.df[tf]["close"])[0])
        #
        #     setattr(self, f"direction_{tf.lower()}", np.zeros_like(self.df[tf]["close"]))
        #
        #     for i in range(len(self.df[tf]["close"])):
        #         if getattr(self, f"hma_slope_{tf.lower()}")[i-1] > 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif getattr(self, f"hma_slope_{tf.lower()}")[i-1] < 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if getattr(self, f"tma_slope_{tf.lower()}")[i-1] > 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif getattr(self, f"tma_slope_{tf.lower()}")[i-1] < 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if getattr(self, f"tsv_{tf.lower()}")[i - 1] > 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif getattr(self, f"tsv_{tf.lower()}")[i - 1] < 0:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if self.df[tf]["close"][i-1] > getattr(self, f"ichimoku_{tf.lower()}")[0][i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif self.df[tf]["close"][i-1] < getattr(self, f"ichimoku_{tf.lower()}")[0][i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if self.df[tf]["close"][i-1] > getattr(self, f"ichimoku_{tf.lower()}")[1][i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif self.df[tf]["close"][i-1] < getattr(self, f"ichimoku_{tf.lower()}")[1][i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if getattr(self, f"wr_21_{tf.lower()}")[i - 2] > -80:
        #             if getattr(self, f"wr_21_{tf.lower()}")[i - 1] <= -80:
        #                 getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif getattr(self, f"wr_21_{tf.lower()}")[i - 2] < -20:
        #             if getattr(self, f"wr_21_{tf.lower()}")[i - 2] >= -20:
        #                 getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #         if self.df[tf]["close"][i-1] > getattr(self, f"supertrend_{tf.lower()}")[i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] += 1
        #         elif self.df[tf]["close"][i-1] < getattr(self, f"supertrend_{tf.lower()}")[i - 1]:
        #             getattr(self, f"direction_{tf.lower()}")[i] -= 1
        #
        #     if tf != self.tfs_list[-1]:
        #         setattr(self, f"direction_{tf.lower()}", match_indexes(
        #             self.df[self.tfs_list[-1]]["time"], self.df[tf]["time"],
        #             getattr(self, f"direction_{tf.lower()}")
        #         ))

        # self.direction = np.zeros_like(self.close)
        #
        # for tf in self.tfs_list:
        #     # if tf != "M1" or tf != "M5":
        #     self.direction += getattr(self, f"direction_{tf.lower()}")
        #
        # self.direction = ema(self.direction, 200)

        # self.h1_hma = hma(self.h1_close, 55)
        # self.h1_hma = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.h1_hma
        # )
        # self.d1_hma = hma(self.d1_close, 55)
        # self.d1_hma = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["D1"]["data"]["time"], self.d1_hma
        # )
        # self.hma_slope = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.hma_slope
        # )

        # self.tsv_histogram, self.tsv_ma = tsv(self.close, self.vol)
        # self.tsv_histogram = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.tsv_histogram
        # )
        # self.tsv_ma = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.tsv_ma
        # )

        # self.rsi = rsi(self.close)
        # self.mfi = mfi(self.high, self.low, self.close, self.vol, n=self.mfi_period)
        # self.small_ema = ema(self.close, period=self.small_ema_period)
        # self.big_ema = ema(self.close, period=self.big_ema_period)
        # self.wr_21 = get_wr(self.high, self.low, self.close)
        # self.wr_21 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.wr_21
        # )
        # self.wr_105 = get_wr(self.high, self.low, self.close, n=105)
        # self.m15_wr_105 = get_wr(self.m15_high, self.m15_low, self.m15_close, n=105)
        # self.m15_wr_21 = get_wr(self.m15_high, self.m15_low, self.m15_close, n=21)
        # self.m15_wr_21 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.m15_index, self.m15_wr_21
        # )
        # self.h1_wr_21 = get_wr(self.h1_high, self.h1_low, self.h1_close, n=21)
        # self.h1_wr_21 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.h1_index, self.h1_wr_21
        # )
        # self.m15_wr_105 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_wr_105
        # )
        # self.concentration_bars, self.three_line_strike = candlestick_patterns(self.open, self.high, self.low,
        #                                                                        self.close)
        # self.trading_allowed = trading_allowed(self.index, 7, 17)
        # self.trading_allowed = trading_allowed(self.h1_index, 7, 22)
        # self.trading_allowed = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.trading_allowed
        # )
        # self.senkou_span_a, self.senkou_span_b = get_ichimoku(self.high, self.low)
        # self.senkou_span_a, self.senkou_span_b = get_ichimoku(self.m15_high, self.m15_low)
        # self.senkou_span_a = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.senkou_span_a
        # )
        # self.senkou_span_b = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.senkou_span_b
        # )

        # self.d1_tema_50 = tema(self.d1_close, 50)
        # self.d1_tema_slope = slope(tema(self.d1_close, 50), 2)
        # self.m15_ema_50 = ema(self.m15_close, 50)
        # self.m15_ema_50 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_ema_50
        # )
        # self.h1_ema_50 = ema(self.h1_close, 50)
        # self.h1_ema_50 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.h1_ema_50
        # )
        # self.d1_tema_50 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_tema_50
        # )
        # self.d1_tema_slope = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_tema_slope
        # )
        # self.macd_line, self.macd_signal, self.macd_histogram = macd(self.close, self.fast_ema_period,
        #                                                              self.slow_ema_period, self.signal_ema_period)

        # self.ema_25 = ema(self.close, 25)
        # self.ema_50 = ema(self.close, 50)
        # self.ema_100 = ema(self.close, 100)
        # self.ema_20_slope = slope(self.ema_20, 1)
        # self.ema_20 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["M15"]["data"]["time"], self.ema_20
        # )
        # self.ema_20_slope = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["M15"]["data"]["time"], self.ema_20_slope
        # )
        # self.ema_200 = ema(self.close, 200)
        # self.ema_200 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.ema_200
        # )
        # self.tema_200 = tema(self.close, 200)
        # self.ema_200_slope = slope(self.ema_200, 1)
        # self.tema_200_slope = slope(self.tema_200, 1)
        # self.entry_tema = tema(self.df["M15"]["data"]["close"], self.entry_tema_period)
        # self.entry_ema = ema(self.df["M15"]["data"]["close"], self.entry_ema_period)
        # self.exit_tema = tema(self.df["M15"]["data"]["close"], self.exit_tema_period)
        # self.exit_ema = ema(self.df["M15"]["data"]["close"], self.exit_ema_period)
        # self.ema_50 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.ema_50
        # )
        # self.ema_10 = ema(self.d1_close, 10)
        # self.ema_10 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["D1"]["time"], self.ema_10
        # )
        # self.ema_50 = ema(self.d1_close, 50)
        # self.ema_50 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["D1"]["time"], self.ema_50
        # )
        # self.ema_200 = ema(self.d1_close, 200)
        # self.ema_200 = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["D1"]["time"], self.ema_200
        # )

        # self.fast_stoch, self.slow_stoch = stochastic(self.close)
        # self.fast_stoch = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_stoch
        # )
        # self.slow_stoch = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_stoch
        # )

        # self.fast_stoch_rsi, self.slow_stoch_rsi = stochastic(rsi(self.close))

        # self.fast_stoch_rsi = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_stoch_rsi
        # )
        # self.slow_stoch_rsi = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.slow_stoch_rsi
        # )

        # for tf in self.strategy_dict["tfs"]:
        #     wpr_trend_data = wpr_trend(
        #         get_wr(self.df[tf]["data"]["high"], self.df[tf]["data"]["low"], self.df[tf]["data"]["close"]),
        #         self.df[tf]["data"]["close"], 21, self.multiplier)
        #
        #     self.df[tf].update(
        #         {}
        #     )

        # self.trend, self.classic_trend, self.concentration, self.strength, self.wpr_up_lines, self.wpr_down_lines = \
        #     wpr_trend(get_wr(self.high, self.low, self.close), self.high, self.low, self.close, 21, self.multiplier)

        # self.strength = slope(self.strength, 10)

        # self.m5_trend, self.m5_classic_trend, self.m5_concentration, self.m5_strength, self.m5_wpr_up_lines, \
        #     self.m5_wpr_down_lines = wpr_trend(
        #         get_wr(self.m5_high, self.m5_low, self.m5_close), self.m5_close, 21, self.multiplier
        #     )
        # self.m5_trend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m5_index, self.m5_trend
        # )

        # self.m15_trend, self.m15_classic_trend, self.m15_concentration, self.m15_strength, self.m15_wpr_up_lines, \
        # self.m15_wpr_down_lines = wpr_trend(
        #     get_wr(self.m15_high, self.m15_low, self.m15_close), self.m15_high, self.m15_low, self.m15_close, 21,
        #     self.multiplier
        # )
        #
        # self.m15_strength = slope(self.m15_strength, 10)
        #
        # self.m15_strength = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_strength
        # )
        #
        # self.h1_trend, self.h1_classic_trend, self.h1_concentration, self.h1_strength, self.h1_wpr_up_lines, \
        #     self.h1_wpr_down_lines = wpr_trend(
        #         get_wr(self.h1_high, self.h1_low, self.h1_close), self.h1_high, self.h1_low, self.h1_close, 21,
        #         self.multiplier
        #     )
        #
        # self.h1_strength = slope(self.h1_strength, 10)
        #
        # self.h1_strength = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.h1_strength
        # )
        #
        # self.h4_trend, self.h4_classic_trend, self.h4_concentration, self.h4_strength, self.h4_wpr_up_lines, \
        # self.h4_wpr_down_lines = wpr_trend(
        #     get_wr(self.h4_high, self.h4_low, self.h4_close), self.h4_high, self.h4_low, self.h4_close, 21,
        #     self.multiplier
        # )
        #
        # self.h4_strength = slope(self.h4_strength, 10)
        #
        # self.h4_strength = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h4_index, self.h4_strength
        # )

        # self.d1_trend, self.d1_classic_trend, self.d1_concentration, self.d1_strength, self.d1_wpr_up_lines, \
        # self.d1_wpr_down_lines = wpr_trend(
        #     get_wr(self.d1_high, self.d1_low, self.d1_close), self.d1_high, self.d1_low, self.d1_close, 21,
        #     self.multiplier
        # )
        #
        # self.d1_strength = slope(self.d1_strength, 10)
        #
        # self.d1_strength = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_strength
        # )

        # self.h1_trend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.h1_trend
        # )

        # self.trading_range = slope(ema(rolling_max(self.high, 600) - rolling_min(self.low, 600), 20), 1)

        # self.macd_line, self.macd_signal, self.macd_histogram = macd(self.close)

        # self.macd_line, self.macd_signal, self.macd_histogram = macd(self.df["M1"]["data"]['close'])

        # self.macd_line = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.macd_line,
        #     shift_to_match=True
        # )
        # self.macd_signal = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.macd_signal,
        #     shift_to_match=True
        # )
        # self.macd_histogram = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df["H1"]["data"]["time"], self.macd_histogram,
        #     shift_to_match=True
        # )

        # self.overlay_open = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["M30"]["time"],
        #     self.df["M30"]["open"]
        # )
        # self.overlay_high = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["M30"]["time"],
        #     self.df["M30"]["high"]
        # )
        # self.overlay_low = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["M30"]["time"],
        #     self.df["M30"]["low"]
        # )
        # self.overlay_close = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.df["M30"]["time"],
        #     self.df["M30"]["close"]
        # )

        # self.dema = dema(self.close, 200)

        # self.atr = atr(self.high, self.low, self.close, self.atr_period)
        # self.atr_upper = self.close + self.atr * self.atr_multiplier
        # self.atr_lower = self.close - self.atr * self.atr_multiplier

        # self.tma()

        # self.df["M1"]["middle_line_tema_slope"]

        # self.tma_direction = np.zeros_like(self.close)
        #
        # for i in range(1, len(self.close)):
        #     direction = 0
        #     for tf in self.strategy_dict["tfs"]:
        #         if self.df[tf]["middle_line_tema_slope"][i] > 0:
        #             if self.df[tf]["middle_line_tema_slope"][i-1] >= self.df[tf]["middle_line_tema_slope"][i]:
        #                 direction += 1
        #         elif self.df[tf]["middle_line_tema_slope"][i] < 0:
        #             if self.df[tf]["middle_line_tema_slope"][i-1] <= self.df[tf]["middle_line_tema_slope"][i]:
        #                 direction -= 1
        #     if direction == len(self.strategy_dict["tfs"]):
        #         self.tma_direction[i] = 1
        #     elif direction == -len(self.strategy_dict["tfs"]):
        #         self.tma_direction[i] = -1

        # self.m5_tma_middle = indicators.tma(self.m5_high, self.m5_low, self.m5_close, 20)[1]
        # self.m5_tma_slope = indicators.slope(self.m5_tma_middle, self.tma_slope_period)
        # self.m5_tma_slope = indicators.match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.m5_index, self.m5_tma_slope
        # )
        # self.m15_tma_middle = tma(self.m15_high, self.m15_low, self.m15_close, 20)[1]
        # self.m15_tma_slope = slope(self.m15_tma_middle, 2)
        # self.m15_tma_slope = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_tma_slope
        # )

        # self.m1_tma_width = self.df["M1"]["tma"]["upper_line"] - self.df["M1"]["tma"]["middle_line"]
        # self.tma_upper = self.df["M1"]["tma"]["upper_line"]
        # self.tma_middle = self.df["M1"]["tma"]["middle_line"]
        # self.tma_lower = self.df["M1"]["tma"]["lower_line"]
        # self.h1_tma_upper = self.df["H1"]["tma"]["upper_line"]
        # self.h1_tma_middle = self.df["H1"]["tma"]["middle_line"]
        # self.h1_tma_lower = self.df["H1"]["tma"]["lower_line"]
        # self.m5_tma_upper = self.df["M5"]["tma"]["upper_line"]
        # self.m5_tma_lower = self.df["M5"]["tma"]["lower_line"]
        # self.m15_tma_upper = self.df["M15"]["tma"]["upper_line"]
        # self.m15_tma_lower = self.df["M15"]["tma"]["lower_line"]
        # self.h1_tma_upper = self.df["H1"]["tma"]["upper_line"]
        # self.h1_tma_lower = self.df["H1"]["tma"]["lower_line"]

        # self.tma_slope = self.df["M1"]["middle_line_tema_slope"]
        # self.m15_tma_slope = self.df["M15"]["middle_line_tema_slope"]
        # self.h1_tma_slope = self.df["H1"]["middle_line_tema_slope"]
        # self.tma_slope_upper = slope(self.tma_upper, 1)
        # self.tma_slope_lower = slope(self.tma_lower, 1)
        # self.h1_tma_slope = tema(slope(self.h1_tma_middle, 1), 5)
        # self.h1_tma_slope_upper = slope(self.h1_tma_upper, 1)
        # self.h1_tma_slope_lower = slope(self.h1_tma_lower, 1)

        # self.mml = mml_calculator(self.m15_high, self.m15_low)
        # self.mml = [match_indexes(self.df[self.strategy_dict["tfs"][0]]["data"]["time"],
        #                           self.df["M15"]["data"]["time"], level) for level in self.mml]

        # self.smma_21 = smma(self.close, 21)
        # self.smma_50 = smma(self.close, 50)
        # self.smma_200 = smma(self.close, 200)
        # self.wills_fractals = indicators.wills_fractals(self.high, self.low)

        # self.lower_line, self.middle_line, self.upper_line = tma(self.high, self.low, self.close, 20)

        # self.irb = inventory_retracement_bar(self.m15_open, self.m15_high, self.m15_low, self.m15_close)
        # self.irb = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.irb
        # )

        # self.slow_supertrend, self.slow_superuptrend, self.slow_superdowntrend = supertrend(self.high, self.low,
        #                                                                                     self.close, 12, 3)
        # self.med_supertrend, self.med_superuptrend, self.med_superdowntrend = supertrend(self.high, self.low,
        #                                                                                  self.close, 11, 2)
        # self.fast_supertrend, self.fast_superuptrend, self.fast_superdowntrend = supertrend(self.high, self.low,
        #                                                                                     self.close, 10, 1)

        # self.rsi = rsi(self.close)

        # self.m5_supertrend, self.m5_superuptrend, self.m5_superdowntrend = supertrend(self.m5_high, self.m5_low,
        #                                                                            self.m5_close, 12, 3)
        # self.m5_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m5_index, self.m5_supertrend
        # )
        # self.m15_supertrend, self.m15_superuptrend, self.m15_superdowntrend = supertrend(self.m15_high, self.m15_low,
        #                                                                            self.m15_close, 12, 3)
        # self.m15_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_supertrend
        # )
        # self.m30_supertrend, self.m30_superuptrend, self.m30_superdowntrend = supertrend(self.m30_high, self.m30_low,
        #                                                                            self.m30_close, 12, 3)
        # self.m30_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m30_index, self.m30_supertrend
        # )
        # self.h1_supertrend, self.h1_superuptrend, self.h1_superdowntrend = supertrend(self.h1_high, self.h1_low,
        #                                                                               self.h1_close, 12, 3)
        # self.h1_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.h1_index, self.h1_supertrend
        # )
        # self.h1_superuptrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.h1_index, self.h1_superuptrend
        # )
        # self.h1_superdowntrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["time"], self.h1_index, self.h1_superdowntrend
        # )
        # self.d1_supertrend, self.d1_superuptrend, self.d1_superdowntrend = supertrend(self.d1_high, self.d1_low,
        #                                                                            self.d1_close, 12, 3)
        # self.d1_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_supertrend
        # )
        # self.d1_superuptrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_superuptrend
        # )
        # self.d1_superdowntrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.d1_index, self.d1_superdowntrend
        # )
        # self.slow_superuptrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.slow_superuptrend
        # )
        # self.slow_superdowntrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.slow_superdowntrend
        # )
        # self.med_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.med_supertrend
        # )
        # self.med_superuptrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.med_superuptrend
        # )
        # self.med_superdowntrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.med_superdowntrend
        # )
        # self.fast_supertrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_supertrend
        # )
        # self.fast_superuptrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_superuptrend
        # )
        # self.fast_superdowntrend = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.h1_index, self.fast_superdowntrend
        # )

        # self.supertrend, self.superuptrend, self.superdowntrend = supertrend(self.high, self.low, self.close)
        # self.green_supertrend, self.green_superuptrend, self.green_superdowntrend = supertrend(self.high, self.low,
        #                                                                                  self.close,
        #                                                                      multiplier=2)

        # peaks_up, peaks_down = peaks(self.high, self.low)

        # self.high_peaks, self.low_peaks = trend_lines(self.high, self.low, self.close, plotting=True)

        self.take_profit = 1
        self.active_longs = []
        self.active_shorts = []
        self.long_tps = []
        self.short_tps = []
        self.long_sls = []
        self.short_sls = []
        self.long_lot_sizes = []
        self.short_lot_sizes = []
        self.profits = []
        self.losses = []
        self.profit_pips = []
        self.loss_pips = []
        self.went_long_price = np.empty(len(self.high))
        self.closed_long_profit = np.empty(len(self.high))
        self.closed_long_loss = np.empty(len(self.high))
        self.went_short_price = np.empty(len(self.high))
        self.closed_short_profit = np.empty(len(self.high))
        self.closed_short_loss = np.empty(len(self.high))
        self.went_long_price[:] = np.nan
        self.closed_long_profit[:] = np.nan
        self.closed_long_loss[:] = np.nan
        self.went_short_price[:] = np.nan
        self.closed_short_profit[:] = np.nan
        self.closed_short_loss[:] = np.nan
        self.closed_long_idx = []
        self.closed_short_idx = []
        self.pips = 0
        self.profit_trades = 0
        self.loss_trades = 0
        self.profit_longs = 0
        self.loss_longs = 0
        self.profit_shorts = 0
        self.loss_shorts = 0
        self.total_trades = 0
        self.cushion = cushion

        self.exhaustion_up = False
        self.exhaustion_down = False
        self.long_pullback = 0
        self.short_pullback = 0
        self.today_opening = 0
        self.direction_lost = 0
        self.adjust_lot_size = True
        self.ready_long = False
        self.ready_short = False

        self.base_level = round(self.close[0], 3) + 3.1 * self.multiplier
        self.up_level = self.base_level + 10 * self.multiplier
        self.down_level = self.base_level - 10 * self.multiplier
        self.last_level = 0
        self.level_step = 10 * self.multiplier

        self.done_longs = []
        self.done_shorts = []

        self.ready_to_long = False
        self.ready_to_short = False

        self.high_range = 0
        self.low_range = 0
        self.finished = True

        # Psychological Levels
        # if self.multiplier == 0.0001:
        #     level = round(self.close[index - 1], 2)
        # else:
        #     level = round(self.close[index - 1])

        # self.m15_high = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_high
        # )
        # self.m15_low = match_indexes(
        #     self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.m15_index, self.m15_low
        # )

    def tester(self, account, index):
        # Add Profit Percent calculator
        # Add ADX Indicator
        # Move config file to os.env

        self.check_exit(account, index, trailing_stop=True, fixed_tp_sl=False, martin=False, fast_growth=False)

        if len(self.active_longs) == 0 and len(self.active_shorts) == 0:
            if self.direction_lost == -1:
                self.open_long(index, account.lot_size)
            elif self.direction_lost == 1:
                self.open_short(index, account.lot_size)
            else:
                direction = random.randint(-1, 1)
                if direction == 1:
                    self.open_long(index, account.lot_size)
                else:
                    self.open_short(index, account.lot_size)

        # self.check_exit(account, index, fixed_tp_sl=True)
        #
        # if self.h1_supertrend[index-1] == self.h1_superuptrend[index-1]:
        #     if self.h1_wr_21[index-2] > -80:
        #         if self.h1_wr_21[index-1] <= -80:
        #             self.open_long(index, account.lot_size)

            # if self.m15_wr_21[index-2] <= -80:
            #     if self.m15_wr_21[index-1] > -80:
            #         self.open_long(index, account.lot_size)

        # elif self.h1_supertrend[index-1] == self.h1_superdowntrend[index-1]:
        #     if self.h1_wr_21[index-2] < -20:
        #         if self.h1_wr_21[index-1] >= -20:
        #             self.open_short(index, account.lot_size)

            # if self.m15_wr_21[index-2] >= -20:
            #     if self.m15_wr_21[index-1] < -20:
            #         self.open_short(index, account.lot_size)

        # if np.isnan(self.h1_superuptrend[index-2]):
        #     if not np.isnan(self.h1_superuptrend[index-1]):
        #         self.open_long(index, account.lot_size)
        #
        # if np.isnan(self.h1_superdowntrend[index-2]):
        #     if not np.isnan(self.h1_superdowntrend[index-1]):
        #         self.open_short(index, account.lot_size)

        # if self.m15_ema_50[index-1] > self.h1_ema_50[index-1]:
        #     self.open_long(index, account.lot_size)
        #
        # if self.m15_ema_50[index-1] < self.h1_ema_50[index-1]:
        #     self.open_short(index, account.lot_size)

        # self.check_exit(account, index)
        #
        # if self.irb[index-1] == 1:
        #     if self.irb[index-2] != 1:
        #         if self.ema_20[index-1] > self.ema_200[index-1]:
        #             if self.ema_20_slope[index-1] > self.slope_level:
        #                 self.open_long(
        #                     index, account.lot_size, custom_sl_level=self.m15_low[index-16] - 2 * self.multiplier,
        #                     sl_multiplied_tp=self.tp_multiplier
        #                 )
        #
        # if self.irb[index-1] == -1:
        #     if self.irb[index-2] != -1:
        #         if self.ema_20[index-1] < self.ema_200[index-1]:
        #             if self.ema_20_slope[index-1] < -self.slope_level:
        #                 self.open_short(
        #                     index, account.lot_size, custom_sl_level=self.m15_high[index-16] + 2 * self.multiplier,
        #                     sl_multiplied_tp=self.tp_multiplier
        #                 )

        # self.check_exit(account, index)

        # if self.d1_tema_50[index-2] >= self.d1_ema_200[index-2]:
        #     if self.d1_tema_50[index-1] < self.d1_ema_200[index-1]:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.d1_tema_50[index-2] <= self.d1_ema_200[index-2]:
        #     if self.d1_tema_50[index-1] > self.d1_ema_200[index-1]:
        #         self.check_exit(account, index, close_short=True)

        # if len(self.active_longs) == 0 and len(self.active_shorts) == 0:
        # if self.m15_tma_slope[index-1] >= 0:
        #     if self.tma_slope[index-1] >= 0:
        #         if self.close[index-1] - self.base_level >= self.level_step:
        #             self.open_long(index, account.lot_size)
        #             self.base_level = round(self.close[index], 3) + 3.1 * self.multiplier
        #
        # if self.m15_tma_slope[index - 1] <= 0:
        #     if self.tma_slope[index - 1] <= 0:
        #         if self.base_level - self.close[index-1] >= self.level_step:
        #             self.open_short(index, account.lot_size)
        #             self.base_level = round(self.close[index], 3) + 3.1 * self.multiplier
        #
        # if len(self.active_longs) > 0:
        #     if self.base_level - self.close[index-1] >= self.level_step:
        #         self.open_long(index, account.lot_size)
        #         self.base_level = round(self.close[index], 3) + 3.1 * self.multiplier
        #
        # if len(self.active_shorts) > 0:
        #     if self.close[index-1] - self.base_level >= self.level_step:
        #         self.open_short(index, account.lot_size)
        #         self.base_level = round(self.close[index], 3) + 3.1 * self.multiplier

        # self.check_exit(account, index, atr_trailing=True)
        #
        # if self.multiplier == 0.0001:
        #     level = round(self.close[index-1], 2)
        # else:
        #     level = round(self.close[index-1])
        #
        # if level != self.last_level:
        #     if self.low[index-1] <= level:
        #         if self.high[index-1] >= level:
        #             self.open_long(index, account.lot_size)
        #             self.open_short(index, account.lot_size)
        #             self.last_level = level
        #
        # if index == account.bars_len-1:
        #     self.check_exit(account, index, close_long=True)
        #     self.check_exit(account, index, close_short=True)

        # Apply Mean Reversion (MML, TMA M15)
        # And take entries on every WR 21 signal while the Mean Reversion trade is active (2nd step)

        # self.check_exit(account, index)

        # if self.high[index-1] >= self.mml[4][index-1]:
        #     self.check_exit(account, index, close_long=True)
        #
        # if self.low[index-1] <= self.mml[4][index-1]:
        #     self.check_exit(account, index, close_short=True)

        # if self.m15_strength[index-1] > 0:
        #     if self.wr_21[index-2] >= -80:
        #         if self.wr_21[index-1] < -80:
        #             self.open_long(index, account.lot_size)
        #
        # if self.m15_strength[index-1] < 0:
        #     if self.wr_21[index-2] <= -20:
        #         if self.wr_21[index-1] > -20:
        #             self.open_short(index, account.lot_size)

        # self.check_exit(account, index)

        # if self.trend[index-2] != 1:
        #     if self.trend[index-1] == 1:
        #         self.open_long(index, account.lot_size)
        #
        # if self.trend[index-2] != -1:
        #     if self.trend[index-1] == -1:
        #         self.open_short(index, account.lot_size)
        #
        # if self.trend[index-2] == 1:
        #     if self.trend[index-1] != 1:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.trend[index-2] == -1:
        #     if self.trend[index-1] != -1:
        #         self.check_exit(account, index, close_short=True)

        # self.check_exit(account, index)
        #
        # if self.concentration[index-1] == 1:
        #     self.open_long(index, account.lot_size)
        # elif self.concentration[index-1] == -1:
        #     self.open_short(index, account.lot_size)

        # if self.entry_tema[index - 2] <= self.entry_ema[index - 2]:
        #     if self.entry_tema[index - 1] > self.entry_ema[index - 1]:
        #         self.open_long(index, account.lot_size)
        #
        # if self.entry_tema[index - 2] >= self.entry_ema[index - 2]:
        #     if self.entry_tema[index - 1] < self.entry_ema[index - 1]:
        #         self.open_short(index, account.lot_size)
        #
        # if self.exit_tema[index-2] >= self.exit_ema[index-2]:
        #     if self.exit_tema[index-1] < self.exit_ema[index-1]:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.exit_tema[index-2] <= self.exit_ema[index-2]:
        #     if self.exit_tema[index-1] > self.exit_ema[index-1]:
        #         self.check_exit(account, index, close_short=True)

        # if self.ema_50[index-1] > self.ema_200[index-1]:
        #     if self.wr_21[index-2] <= -80:
        #         if self.wr_21[index-1] > -80:
        #             self.open_long(index, account.lot_size)
        #
        #     if self.wr_21[index-1] >= -20:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.ema_50[index-1] < self.ema_200[index-1]:
        #     if self.wr_21[index-2] >= -20:
        #         if self.wr_21[index-1] < -20:
        #             self.open_short(index, account.lot_size)
        #
        #     if self.wr_21[index-1] <= -80:
        #         self.check_exit(account, index, close_short=True)

        # if self.ema_50[index-2] > self.ema_200[index-2]:
        #     if self.ema_50[index-1] <= self.ema_200[index-1]:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.ema_50[index-2] < self.ema_200[index-2]:
        #     if self.ema_50[index-1] >= self.ema_200[index-1]:
        #         self.check_exit(account, index, close_short=True)

        # if self.trend[index-2] == 1:
        #     if self.trend[index-1] != 1:
        #         self.check_exit(account, index, close_long=True)
        #
        # if self.trend[index-2] == -1:
        #     if self.trend[index-1] != -1:
        #         self.check_exit(account, index, close_short=True)

    def h1_mml(self, account, index):
        if self.close[index - 1] >= self.mml[6][index - 1]:
            self.check_exit(account, index, close_long=True)
            self.check_exit(account, index, close_short=True)

        if self.close[index - 1] <= self.mml[2][index - 1]:
            self.check_exit(account, index, close_short=True)
            self.check_exit(account, index, close_long=True)

        if self.tma_slope[index - 1] >= 0:
            if self.low[index - 2] <= self.mml[4][index - 2]:
                if self.high[index - 2] >= self.mml[4][index - 2]:
                    if self.close[index - 2] < self.open[index - 2]:
                        if self.close[index - 1] > self.mml[4][index - 1]:
                            if self.close[index - 1] > self.open[index - 1]:
                                self.open_long(index, account.lot_size)

        if self.tma_slope[index - 1] <= 0:
            if self.high[index - 2] >= self.mml[4][index - 2]:
                if self.low[index - 2] <= self.mml[4][index - 2]:
                    if self.close[index - 2] > self.open[index - 2]:
                        if self.close[index - 1] < self.mml[4][index - 1]:
                            if self.close[index - 1] < self.open[index - 1]:
                                self.open_short(index, account.lot_size)

    def hunter_2(self, account, index):
        self.check_exit(account, index)

        if self.trend[index - 1] == 1:
            if self.wr_105[index - 1] >= -50:
                if self.strength[index - 1] >= 0:
                    if self.close[index - 1] > self.senkou_span_a[index - 1]:
                        if self.close[index - 1] > self.senkou_span_b[index - 1]:
                            if np.sum(self.m1_tma_slope_lower[index - 6:index - 1]) > 0:
                                if self.m1_tma_slope_lower[index - 1] > 0:
                                    if self.wr_21[index - 2] <= -80:
                                        if self.wr_21[index - 1] > -80:
                                            self.open_long(index, account.lot_size)

        elif self.trend[index - 1] == -1:
            if self.wr_105[index - 1] <= -50:
                if self.strength[index - 1] <= 0:
                    if self.close[index - 1] < self.senkou_span_a[index - 1]:
                        if self.close[index - 1] < self.senkou_span_b[index - 1]:
                            if np.sum(self.m1_tma_slope_upper[index - 6:index - 1]) < 0:
                                if self.m1_tma_slope_upper[index - 1] < 0:
                                    if self.wr_21[index - 2] >= -20:
                                        if self.wr_21[index - 1] < -20:
                                            self.open_short(index, account.lot_size)

    def hunter(self, account, index):
        # Requires to be trained on 400-700 days worth of data

        # Add TMA Width Slope (period is to be determined by the GA) to check if the width is currently growing

        # self.tp = round(self.params[0])
        # self.sl = round(self.params[1])
        # self.atr_period = round(self.params[2])
        # self.tma_slope_period = round(self.params[3])
        # self.m1_tma_slope_level_long = round(self.params[4], 6)
        # self.m1_tma_slope_level_short = round(self.params[5], 6)
        # self.m1_tma_core_entry_slope = round(self.params[6], 6)
        # self.m1_tma_pattern_entry_slope = round(self.params[7], 6)
        # self.m1_tma_width_pattern_entry = round(self.params[8], 6)
        # self.m1_tma_width_core_entry = round(self.params[9], 6)
        # self.m1_tma_width_entry_level = round(self.params[10], 6)
        # self.strength_period = round(self.params[11])
        # self.strength_level_long = round(self.params[12], 3)
        # self.strength_level_short = round(self.params[13], 3)
        # self.m5_tma_slope_level = round(self.params[14], 6)
        # self.atr_multiplier = round(self.params[15])
        # self.use_exhaustion = round(self.params[16])
        # self.max_previous_bar_size = round(self.params[17])
        # self.start_hour = round(self.params[18])
        # self.end_hour = round(self.params[19])
        # self.m1_tma_slope_long_slow_stop = round(self.params[20], 6)
        # self.m1_tma_slope_long_medium_stop = round(self.params[21], 6)
        # self.m1_tma_slope_long_fast_stop = round(self.params[22], 6)
        # self.m1_tma_slope_short_slow_stop = round(self.params[23], 6)
        # self.m1_tma_slope_short_medium_stop = round(self.params[24], 6)
        # self.m1_tma_slope_short_fast_stop = round(self.params[25], 6)

        self.check_exit(account, index, martin=False, atr_trailing=True, fast_growth=False)

        if self.use_exhaustion == 1:
            if self.m1_tma_upper[index - 1] >= self.m15_tma_upper[index - 1]:
                self.exhaustion_up = True
            elif self.m1_tma_lower[index - 1] <= self.m15_tma_lower[index - 1]:
                self.exhaustion_down = True

            if self.exhaustion_up:
                if self.trend[index - 2] != 1 and self.trend[index - 1] == 1:
                    self.exhaustion_up = False
            if self.exhaustion_down:
                if self.trend[index - 2] != -1 and self.trend[index - 1] == -1:
                    self.exhaustion_down = False

        if self.trading_allowed[index]:
            if self.m1_tma_width[index - 1] >= self.m1_tma_width_entry_level:
                if abs(self.open[index - 1] - self.close[index - 1]) / self.multiplier < self.max_previous_bar_size:

                    if self.m15_tma_slope[index - 1] >= self.m15_tma_slope[index - 16]:
                        if self.m5_tma_slope[index - 1] >= self.m5_tma_slope[index - 6]:
                            if self.strength[index - 1] > self.strength_level_long:
                                if self.m1_tma_slope[index - 1] >= self.m1_tma_slope_level_long:

                                    # Classic Entry
                                    if self.classic_trend[index - 1] == 1:
                                        if not self.exhaustion_up:
                                            if self.wr_105[index - 1] > -50:
                                                if self.m1_tma_middle[index - 1] > self.senkou_span_a[index - 1]:
                                                    if self.m5_tma_slope[index - 1] >= self.m5_tma_slope_level * -1:
                                                        if self.wr_21[index - 2] <= -80 < self.wr_21[index - 1]:
                                                            self.open_long(index, account.lot_size)

                                    # Core Entry
                                    if self.trend[index - 1] == 1:
                                        if self.m1_tma_slope[index - 1] > self.m1_tma_core_entry_slope:
                                            if self.low[index - 1] <= self.m1_tma_lower[index - 1] < self.close[
                                                index - 1]:
                                                self.open_long(index, account.lot_size)
                                            elif self.m1_tma_width[index - 1] >= self.m1_tma_width_core_entry:
                                                if self.low[index - 1] <= self.m1_tma_middle[index - 1] < self.close[
                                                    index - 1]:
                                                    self.open_long(index, account.lot_size)

                                    # Pattern Entry
                                    if self.trend[index - 1] == 1 and self.classic_trend[index - 1] == 1:
                                        if self.m1_tma_lower[index - 1] > self.senkou_span_a[index - 1] and \
                                                self.m1_tma_lower[index - 1] > self.senkou_span_b[index - 1]:
                                            if self.m1_tma_slope[index - 1] >= self.m1_tma_pattern_entry_slope:
                                                if self.m1_tma_width[index - 1] >= self.m1_tma_width_pattern_entry:
                                                    if self.wpr_up_lines[index - 1] and self.wpr_up_lines[index - 2] and \
                                                            self.wpr_up_lines[index - 3]:
                                                        if self.wpr_up_lines[index - 3] == self.wpr_up_lines[
                                                            index - 2] and \
                                                                self.wpr_up_lines[index - 1] > self.wpr_up_lines[
                                                            index - 2]:
                                                            self.open_long(index, account.lot_size)
                                                    elif not self.wpr_up_lines[index - 2] and self.wpr_up_lines[
                                                        index - 1]:
                                                        self.open_long(index, account.lot_size)

                    elif self.m15_tma_slope[index - 1] <= self.m15_tma_slope[index - 16]:
                        if self.m5_tma_slope[index - 1] <= self.m5_tma_slope[index - 6]:
                            if self.strength[index - 1] < self.strength_level_short:
                                if self.m1_tma_slope[index - 1] <= self.m1_tma_slope_level_short:

                                    # Classic Entry
                                    if self.classic_trend[index - 1] == -1:
                                        if not self.exhaustion_down:
                                            if self.wr_105[index - 1] < -50:
                                                if self.m1_tma_middle[index - 1] < self.senkou_span_a[index - 1]:
                                                    if self.m5_tma_slope[index - 1] <= self.m5_tma_slope_level:
                                                        if self.wr_21[index - 2] >= -20 > self.wr_21[index - 1]:
                                                            self.open_short(index, account.lot_size)

                                    # Core Entry
                                    if self.trend[index - 1] == -1:
                                        if self.m1_tma_slope[index - 1] < self.m1_tma_core_entry_slope * -1:
                                            if self.high[index - 1] >= self.m1_tma_upper[index - 1] > self.close[
                                                index - 1]:
                                                self.open_short(index, account.lot_size)
                                            elif self.m1_tma_width[index - 1] >= self.m1_tma_width_core_entry:
                                                if self.high[index - 1] >= self.m1_tma_middle[index - 1] > self.close[
                                                    index - 1]:
                                                    self.open_short(index, account.lot_size)

                                    # Pattern Entry
                                    if self.trend[index - 1] == -1 and self.classic_trend[index - 1] == -1:
                                        if self.m1_tma_upper[index - 1] < self.senkou_span_a[index - 1] and \
                                                self.m1_tma_upper[index - 1] < self.senkou_span_b[index - 1]:
                                            if self.m1_tma_slope[index - 1] <= self.m1_tma_pattern_entry_slope * -1:
                                                if self.m1_tma_width[index - 1] >= self.m1_tma_width_pattern_entry:
                                                    if self.wpr_down_lines[index - 1] and self.wpr_down_lines[
                                                        index - 2] and \
                                                            self.wpr_down_lines[index - 3]:
                                                        if self.wpr_down_lines[index - 3] == self.wpr_down_lines[
                                                            index - 2] and \
                                                                self.wpr_down_lines[index - 1] < self.wpr_down_lines[
                                                            index - 2]:
                                                            self.open_short(index, account.lot_size)
                                                    elif not self.wpr_down_lines[index - 2] and self.wpr_down_lines[
                                                        index - 1]:
                                                        self.open_short(index, account.lot_size)

        if self.track_exits == 1:
            if self.m1_tma_slope[index - 1] < self.m1_tma_slope_long_stop or \
                    self.m1_tma_width[index - 1] < self.m1_tma_width_exit_level:
                self.check_exit(account, index, close_long=True)
            if self.m1_tma_slope[index - 1] > self.m1_tma_slope_short_stop or \
                    self.m1_tma_width[index - 1] < self.m1_tma_width_exit_level:
                self.check_exit(account, index, close_short=True)

        if self.high[index - 1] >= self.m1_tma_upper[index - 1]:
            if self.m1_tma_slope[index - 1] < self.m1_tma_slope_long_slow_stop:
                self.check_exit(account, index, close_long=True)
        elif self.high[index - 1] >= self.m5_tma_upper[index - 1]:
            if self.m1_tma_slope[index - 1] < self.m1_tma_slope_long_medium_stop:
                self.check_exit(account, index, close_long=True)
        elif self.high[index - 1] >= self.m15_tma_upper[index - 1]:
            if self.m1_tma_slope[index - 1] < self.m1_tma_slope_long_fast_stop:
                self.check_exit(account, index, close_long=True)
        elif self.low[index - 1] <= self.m1_tma_lower[index - 1]:
            if self.m1_tma_slope[index - 1] > self.m1_tma_slope_short_slow_stop:
                self.check_exit(account, index, close_short=True)
        elif self.low[index - 1] <= self.m5_tma_lower[index - 1]:
            if self.m1_tma_slope[index - 1] > self.m1_tma_slope_short_medium_stop:
                self.check_exit(account, index, close_short=True)
        elif self.low[index - 1] <= self.m15_tma_lower[index - 1]:
            if self.m1_tma_slope[index - 1] > self.m1_tma_slope_short_fast_stop:
                self.check_exit(account, index, close_short=True)

    def ema_crossing(self, account, index):
        # H1, requires GA, SL 20
        self.check_exit(account, index)

        if self.entry_ema[index - 2] <= self.ema_200[index - 2]:
            if self.entry_ema[index - 1] > self.ema_200[index - 1]:
                self.check_exit(account, index, close_short=True)
                self.open_long(index, account.lot_size)
        if self.entry_ema[index - 2] >= self.ema_200[index - 2]:
            if self.entry_ema[index - 1] < self.ema_200[index - 1]:
                self.check_exit(account, index, close_long=True)
                self.open_short(index, account.lot_size)

    def m30_macd_ema(self, account, index):
        self.check_exit(account, index, martin=False, atr_trailing=True, fast_growth=False)
        # TP 30
        # SL -20
        # ATR_MULTIPLIER 5

        # Predict the next bar to catch MACD LINE crossing over MACD SIGNAL using ML

        # if self.macd_line[index-31] <= 0 and self.macd_line[index-30] > 0:
        #     self.open_long(index, account.lot_size)
        # elif self.macd_line[index-31] >= 0 and self.macd_line[index-30] < 0:
        #     self.open_short(index, account.lot_size)

        if self.high[index - 1] > self.ema_200[index - 1]:
            if self.macd_line[index - 1] < 0 and self.macd_signal[index - 1] < 0:
                if self.macd_line[index - 2] < self.macd_signal[index - 2] and self.macd_line[index - 1] > \
                        self.macd_signal[index - 1]:
                    self.open_long(index, account.lot_size)
        elif self.low[index - 1] < self.ema_200[index - 1]:
            if self.macd_line[index - 1] > 0 and self.macd_signal[index - 1] > 0:
                if self.macd_line[index - 2] > self.macd_signal[index - 2] and self.macd_line[index - 1] < \
                        self.macd_signal[index - 1]:
                    self.open_short(index, account.lot_size)

    def tmt(self, account, index):
        # if self.index[index-1].day != self.index[index-2].day:
        #     self.today_opening = self.open[index-1]

        if self.today_opening != 0:
            current_state = self.open[index] - self.today_opening

            self.check_exit(account, index, atr_trailing=True)

            # if self.trading_allowed[index]:
            if current_state > 0:
                if self.rsi[index - 1] > self.long_rsi_level:
                    if self.mfi[index - 1] > self.long_mfi_level:
                        if self.small_ema[index - 1] > self.big_ema[index - 1]:
                            if self.low[index - 1] < self.small_ema[index - 1] and \
                                    self.low[index - 1] < self.big_ema[index - 1]:
                                self.long_pullback = 1
                            if self.long_pullback == 1:
                                if self.wr[index - 2] < self.long_wr_level <= self.wr[index - 1]:
                                    self.long_pullback = 0
                                    self.open_long(index, account.lot_size)
            elif current_state < 0:
                if self.rsi[index - 1] < self.short_rsi_level:
                    if self.mfi[index - 1] < self.short_mfi_level:
                        if self.small_ema[index - 1] < self.big_ema[index - 1]:
                            if self.high[index - 1] > self.small_ema[index - 1] and \
                                    self.high[index - 1] > self.big_ema[index - 1]:
                                self.short_pullback = 1
                            if self.short_pullback == 1:
                                if self.wr[index - 2] > self.short_wr_level >= self.wr[index - 1]:
                                    self.short_pullback = 0
                                    self.open_short(index, account.lot_size)

    def concentration_martin(self, account, index):
        self.check_exit(account, index, martin=False, atr_trailing=True)

        if len(self.active_shorts) == 0 and len(self.active_longs) == 0:
            if self.concentration_bars[index - 1] == 1:
                self.open_long(index)

            elif self.concentration_bars[index - 1] == -1:
                self.open_short(index)

    def random_martin(self, account, index):
        self.check_exit(account, index, martin=True, atr_trailing=False, fast_growth=False)

        # if self.df.trading_allowed[index] == 1:
        trade_choice = random.randint(0, 50)

        if len(self.active_shorts) == 0 and len(self.active_longs) == 0:
            if account.lot_size > account.lot_size_start:
                if self.direction_lost == 1:
                    self.open_short(index, account.lot_size)
                else:
                    self.open_long(index, account.lot_size)
            else:
                if trade_choice == 1:
                    self.open_long(index, account.lot_size)
                elif trade_choice == 0:
                    self.open_short(index, account.lot_size)

    def open_long(self, index, lot_size, open_price=None, custom_tp=None, custom_sl=None, sl_multiplied_tp=None,
                  previous_tp=False, previous_sl=False):
        if not open_price:
            open_price = self.open[index]

        price = open_price + self.multiplier * (self.spread[index-1] + self.cushion)

        tp = price + self.tp * self.multiplier
        sl = price - self.sl * self.multiplier

        if custom_tp:
            tp = custom_tp

        if custom_sl:
            sl = custom_sl

        if sl_multiplied_tp:
            tp = price + (price - sl) * sl_multiplied_tp

        if previous_tp:
            if len(self.long_tps) > 0:
                self.tp = self.long_tps[-1]

        if previous_sl:
            if len(self.long_sls) > 0:
                self.sl = self.long_sls[-1]

        self.long_tps.append(tp)
        self.long_sls.append(sl)
        self.active_longs.append(price)
        self.long_lot_sizes.append(lot_size)
        self.went_long_price[index] = price

        if self.verbose:
            print(f"Going Long on {self.name}, {lot_size}")
            print(self.index[index])
            print(f"Take Profit: {tp}")
            print(f"Stop Loss: {sl}")
            print("Active Longs: " + str(self.active_longs))
            print("\n")

    def open_short(self, index, lot_size, open_price=None, custom_tp=None, custom_sl=None, sl_multiplied_tp=None,
                   previous_tp=False, previous_sl=False):
        if not open_price:
            open_price = self.open[index]

        tp = open_price - self.tp * self.multiplier
        sl = open_price + self.sl * self.multiplier

        if custom_tp:
            tp = custom_tp

        if custom_sl:
            sl = custom_sl

        if sl_multiplied_tp:
            tp = open_price - (sl - open_price) * sl_multiplied_tp

        if previous_tp:
            if len(self.short_tps) > 0:
                tp = self.short_tps[-1]

        if previous_sl:
            if len(self.short_sls) > 0:
                sl = self.short_sls[-1]

        self.active_shorts.append(open_price)
        self.short_tps.append(tp)
        self.short_sls.append(sl)
        self.short_lot_sizes.append(lot_size)
        self.went_short_price[index] = open_price

        if self.verbose:
            print(f"Going Short on {self.name}, {lot_size}")
            print(self.index[index])
            print(f"Take Profit: {tp}")
            print(f"Stop Loss: {sl}")
            print("Active Shorts: " + str(self.active_shorts))
            print("\n")

    def check_exit(self, account, index, close_long=False, close_short=False, martin=False, break_even=False,
                   trailing_stop=False, atr_trailing=False, fast_growth=False, fixed_tp_sl=False):

        # Add Break Even mode with a fixed tp of ~ 1 pip after price moved away 20+ pips

        if fast_growth:
            self.adjust_lot_size = False
            account.lot_size = account.fast_growth[account.win_series] / 100

        if len(self.active_longs) > 0:
            for i, value in enumerate(self.active_longs):

                current_price = self.open[index]
                pips = 0

                if atr_trailing:
                    if current_price - self.atr[index - 1] * self.atr_multiplier > self.long_sls[i]:
                        self.long_sls[i] = current_price - self.atr[index - 1] * self.atr_multiplier

                if trailing_stop:
                    if current_price - self.trailing_stop * self.multiplier > self.long_sls[i]:
                        self.long_sls[i] = current_price - self.trailing_stop * self.multiplier

                if fixed_tp_sl:
                    if self.high[index-1] >= self.long_tps[i] or current_price >= self.long_tps[i]:
                        pips = round(pip_calc(value, self.long_tps[i], self.multiplier), 2)
                        close_long = True
                    elif self.low[index-1] <= self.long_sls[i] or current_price <= self.long_sls[i]:
                        pips = round(pip_calc(value, self.long_sls[i], self.multiplier), 2)
                        close_long = True

                if current_price >= self.long_tps[i] or current_price <= self.long_sls[i] or close_long:
                    if not fixed_tp_sl:
                        pips = round(pip_calc(value, current_price, self.multiplier), 2)

                    result = round(pips * (self.long_lot_sizes[i] * account.currency_converter), 2)

                    if pips >= 0:
                        if self.verbose:
                            print(f"Closing {self.long_lot_sizes[i]} {self.name} Long in profit")
                            print(f"Price: {current_price}")
                            print('Pips: ' + str(pips))
                        self.profits.append(result)
                        account.profits.append(result)
                        self.profit_pips.append(pips)
                        account.profit_pips.append(pips)
                        self.profit_longs += 1
                        account.profit_longs += 1
                        self.profit_trades += 1
                        account.profit_trades += 1
                        self.closed_long_profit[index] = current_price
                        if martin:
                            account.lot_size = account.lot_size_start
                        self.direction_lost = 0
                        if fast_growth:
                            if account.win_series < len(account.fast_growth) - 1:
                                account.win_series += 1
                    else:
                        if self.verbose:
                            print(f"Closing {self.long_lot_sizes[i]} {self.name} Long in loss")
                            print(f"Price: {current_price}")
                            print('Pips: ' + str(pips))
                        self.losses.append(result)
                        account.losses.append(result)
                        self.loss_pips.append(pips)
                        account.loss_pips.append(pips)
                        self.loss_longs += 1
                        account.loss_longs += 1
                        self.loss_trades += 1
                        account.loss_trades += 1
                        self.closed_long_loss[index] = current_price
                        if martin:
                            account.lot_size *= 2
                        self.direction_lost = 1
                        if fast_growth:
                            if account.win_series > 0:
                                account.win_series -= 1

                    account.balance += result
                    self.pips += pips
                    account.pips += pips
                    self.total_trades += 1
                    account.total_trades += 1

                    self.active_longs.pop(i)
                    self.long_tps.pop(i)
                    self.long_sls.pop(i)
                    self.long_lot_sizes.pop(i)
                    self.closed_long_idx.append(index)
                    if self.verbose:
                        print(f"Result: ${result}")
                        print(f"Balance: ${account.balance}")
                        print(self.index[index])
                        print("Active Longs: " + str(self.active_longs))
                        print("Active Shorts: " + str(self.active_shorts))
                        print("\n")
                    if len(self.active_longs) > 0:
                        self.check_exit(
                            account, index, close_long, close_short, martin, break_even, trailing_stop, atr_trailing,
                            fast_growth, fixed_tp_sl
                        )

        if len(self.active_shorts) > 0:
            for i, value in enumerate(self.active_shorts):

                current_price = self.open[index] + (self.spread[index-1] + self.cushion) * self.multiplier
                pips = 0

                if atr_trailing:
                    if current_price + self.atr[index - 1] * self.atr_multiplier < self.short_sls[i]:
                        self.short_sls[i] = current_price + self.atr[index - 1] * self.atr_multiplier

                if trailing_stop:
                    if current_price + self.trailing_stop * self.multiplier < self.short_sls[i]:
                        self.short_sls[i] = current_price + self.trailing_stop * self.multiplier

                if fixed_tp_sl:
                    if self.low[index-1] + (self.spread[index-1] + self.cushion) * self.multiplier <= self.short_tps[i]\
                            or current_price <= self.short_tps[i]:
                        pips = round((value - self.short_tps[i]) / self.multiplier, 2)
                    elif self.high[index-1] + (self.spread[index-1] + self.cushion) * self.multiplier >= \
                            self.short_sls[i] or current_price >= self.short_sls[i]:
                        pips = round((value - self.short_sls[i]) / self.multiplier, 2)

                if current_price <= self.short_tps[i] or current_price >= self.short_sls[i] or close_short:
                    if not fixed_tp_sl:
                        pips = round((value - current_price) / self.multiplier, 2)

                    result = round(pips * (self.short_lot_sizes[i] * account.currency_converter), 2)

                    if pips >= 0:
                        if self.verbose:
                            print(f"Closing {self.short_lot_sizes[i]} {self.name} Short in profit")
                            print(f"Price: {current_price}")
                            print('Pips: ' + str(pips))
                        self.profits.append(result)
                        account.profits.append(result)
                        self.profit_pips.append(pips)
                        account.profit_pips.append(pips)
                        self.profit_shorts += 1
                        account.profit_shorts += 1
                        self.profit_trades += 1
                        account.profit_trades += 1
                        self.closed_short_profit[index] = current_price
                        if martin:
                            account.lot_size = account.lot_size_start
                        self.direction_lost = 0
                        if fast_growth:
                            if account.win_series < len(account.fast_growth) - 1:
                                account.win_series += 1
                    else:
                        if self.verbose:
                            print(f"Closing {self.short_lot_sizes[i]} {self.name} Short in loss")
                            print(f"Price: {current_price}")
                            print('Pips: ' + str(pips))
                        self.losses.append(result)
                        account.losses.append(result)
                        self.loss_pips.append(pips)
                        account.loss_pips.append(pips)
                        self.loss_shorts += 1
                        account.loss_shorts += 1
                        self.loss_trades += 1
                        account.loss_trades += 1
                        self.closed_short_loss[index] = current_price
                        if martin:
                            account.lot_size *= 2
                        self.direction_lost = -1
                        if fast_growth:
                            if account.win_series > 0:
                                account.win_series -= 1

                    account.balance += result
                    self.pips += pips
                    account.pips += pips
                    self.total_trades += 1
                    account.total_trades += 1

                    self.active_shorts.pop(i)
                    self.short_tps.pop(i)
                    self.short_sls.pop(i)
                    self.short_lot_sizes.pop(i)
                    self.closed_short_idx.append(index)
                    if self.verbose:
                        print(f"Result: ${result}")
                        print(f"Balance: ${account.balance}")
                        print(self.index[index])
                        print("Active Longs: " + str(self.active_longs))
                        print("Active Shorts: " + str(self.active_shorts))
                        print("\n")
                    if len(self.active_shorts) > 0:
                        self.check_exit(
                            account, index, close_long, close_short, martin, break_even, trailing_stop, atr_trailing,
                            fast_growth, fixed_tp_sl
                        )

    def total_report(self, account):
        print("Report for " + self.name)
        if self.profit_trades != 0:
            print("Winning rate: " + str(round(100 / self.total_trades * self.profit_trades)) + "%")
        else:
            print("Winning rate: 0%")
        print("Profit trades: " + str(self.profit_trades))
        print("Loss trades: " + str(self.loss_trades))
        print("Active longs: " + str(len(self.active_longs)))
        print("Active shorts: " + str(len(self.active_shorts)))
        print("Number of longs: " + str(len(self.closed_long_idx)))
        print("Profit longs: " + str(self.profit_longs))
        print("Loss longs: " + str(self.loss_longs))
        print("Number of shorts: " + str(len(self.closed_short_idx)))
        print("Profit shorts: " + str(self.profit_shorts))
        print("Loss shorts: " + str(self.loss_shorts))
        if len(self.profits) > 0:
            print("Biggest profit: $" + str(max(self.profits)))
        else:
            print("Biggest profit: $0")
        if len(self.losses) > 0:
            print("Biggest loss: -$" + str(min(self.losses))[1:])
        else:
            print("Biggest loss: $0")
        if len(self.profits) > 0:
            print("Average profit: $" + str(round(mean(self.profits), 2)))
        else:
            print("Average profit: $0")
        if len(self.losses) > 0:
            print("Average loss: -$" + str(round(mean(self.losses), 2))[1:])
        else:
            print("Average loss: $0")
        if len(self.profit_pips) > 0:
            print("Max pips made: " + str(max(self.profit_pips)))
        else:
            print("Max pips made: 0")
        if len(self.loss_pips) > 0:
            print("Max pips lost: " + str(min(self.loss_pips)))
        else:
            print("Max pips lost: 0")
        if len(self.profit_pips) > 0:
            print("Average pips made: " + str(round(mean(self.profit_pips), 1)))
        else:
            print("Average pips made: 0")
        if len(self.loss_pips) > 0:
            print("Average pips lost: " + str(round(mean(self.loss_pips), 1)))
        else:
            print("Average pip lost: 0")
        print("Total trades: " + str(self.total_trades))
        if self.total_trades != 0:
            print("Trading rate: " + str((account.trading_period[1] - account.trading_period[0]) / self.total_trades)
                  + " minutes")
        else:
            print("Trading rate: 0")
        print("Pips: " + str(self.pips))
        print("\n")

    def tma(self):
        for tf in self.strategy_dict["tfs"][::-1]:
            self.df[tf]["tma"] = {}
            self.df[tf]["tma"]["lower_line"], self.df[tf]["tma"]["middle_line"], self.df[tf]["tma"]["upper_line"] = \
                tma(self.df[tf]["data"]["high"], self.df[tf]["data"]["low"], self.df[tf]["data"]["close"], 20)

            self.df[tf]["middle_line_tema_slope"] = tema(slope(self.df[tf]["tma"]["middle_line"], 1), 5)

            if tf != self.strategy_dict["tfs"][0]:
                for line in self.df[tf]["tma"]:
                    self.df[tf]["tma"][line] = match_indexes(
                        self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df[tf]["data"]["time"],
                        self.df[tf]["tma"][line]
                    )

                self.df[tf]["middle_line_tema_slope"] = match_indexes(
                    self.df[self.strategy_dict["tfs"][0]]["data"]["time"], self.df[tf]["data"]["time"],
                    self.df[tf]["middle_line_tema_slope"]
                )

    def plotter(self, dark=True):
        if dark:
            fplt.foreground = '#fff'
            fplt.background = '#090c0e'
            fplt.odd_plot_background = '#090c0e'
            fplt.candle_bull_color = fplt.candle_bull_body_color = '#0b0'
            fplt.candle_bear_color = '#a23'
            volume_transparency = '6'
        else:
            fplt.foreground = '#444'
            fplt.background = fplt.candle_bull_body_color = '#fff'
            fplt.candle_bull_color = '#380'
            fplt.candle_bear_color = '#c50'
            volume_transparency = 'c'
        fplt.volume_bull_color = fplt.volume_bull_body_color = fplt.candle_bull_color + volume_transparency
        fplt.volume_bear_color = fplt.candle_bear_color + volume_transparency
        fplt.cross_hair_color = fplt.foreground + '8'
        fplt.draw_line_color = '#888'
        fplt.draw_done_color = '#555'

        # window background
        for win in fplt.windows:
            win.setBackground(fplt.background)

        # axis, crosshair, candlesticks, volumes
        axs = [ax for win in fplt.windows for ax in win.axs]
        axs += fplt.overlay_axs
        axis_pen = fplt._makepen(color=fplt.foreground)
        for ax in axs:
            ax.axes['right']['item'].setPen(axis_pen)
            ax.axes['right']['item'].setTextPen(axis_pen)
            ax.axes['bottom']['item'].setPen(axis_pen)
            ax.axes['bottom']['item'].setTextPen(axis_pen)
            if ax.crosshair is not None:
                ax.crosshair.xtext.setColor(fplt.foreground)
                ax.crosshair.ytext.setColor(fplt.foreground)
            for item in ax.items:
                if isinstance(item, fplt.FinPlotItem):
                    isvolume = ax in fplt.overlay_axs
                    if not isvolume:
                        item.colors.update(
                            dict(bull_shadow=fplt.candle_bull_color,
                                 bull_frame=fplt.candle_bull_color,
                                 bull_body=fplt.candle_bull_body_color,
                                 bear_shadow=fplt.candle_bear_color,
                                 bear_frame=fplt.candle_bear_color,
                                 bear_body=fplt.candle_bear_color))
                    else:
                        item.colors.update(
                            dict(bull_frame=fplt.volume_bull_color,
                                 bull_body=fplt.volume_bull_body_color,
                                 bear_frame=fplt.volume_bear_color,
                                 bear_body=fplt.volume_bear_color))
                    item.repaint()

        fplt.display_timezone = timezone.utc
        x = self.index
        marker_size = 3
        ax1 = fplt.create_plot(self.name, rows=1, init_zoom_periods=200)

        def plotting_function(name, ax=ax1, colors: dict = None):
            if not colors:
                colors = self.colors

            for tf in self.tfs_dict[name]:
                data = getattr(self, name + "_" + tf.lower())
                result = list()

                if tf != self.strategy_dict["tfs"][0]:
                    for i in range(len(data)):
                        result.append(match_indexes(
                            self.df[self.strategy_dict["tfs"][0]]["time"], self.df[tf]["time"], data[i]
                        ))
                else:
                    result = data

                plotted_data = [fplt.plot(x, line, color=colors[tf], ax=ax) for line in result]

                if name == "ichimoku_cloud":
                    fplt.fill_between(plotted_data[0], plotted_data[1], color=colors[tf])

        def plot_supertrend(name='supertrend'):
            for tf in self.tfs_dict[name]:
                data = getattr(self, name + "_" + tf.lower())
                result = list()

                if tf != self.strategy_dict["tfs"][0]:
                    for i in range(len(data)):
                        result.append(match_indexes(
                            self.df[self.strategy_dict["tfs"][0]]["time"], self.df[tf]["time"], data[i]
                        ))
                else:
                    result = data

                fplt.plot(x, result[0], color='blue', ax=ax1)
                fplt.plot(x, result[1], color='green', ax=ax1)
                fplt.plot(x, result[2], color='red', ax=ax1)

        def plot_atr(ax):
            fplt.plot(x, self.atr_upper, legend="Upper ATR", style="...", color="blue", ax=ax)
            fplt.plot(x, self.atr_lower, legend="Lower ATR", style="...", color="blue", ax=ax)

        def plot_wr(ax, period, levels: list, tfs):
            fplt.set_y_range(-100, 0, ax=ax)

            for i in range(len(levels)):
                fplt.add_line((0, levels[i]), (len(x), levels[i]), color='#fff', style='---', ax=ax)

            for tf in tfs:
                wr = get_wr(self.df[tf]["high"], self.df[tf]["low"], self.df[tf]["close"],
                            period)

                if tf != self.strategy_dict["tfs"][0]:
                    wr = match_indexes(
                        self.df[self.strategy_dict["tfs"][0]]["time"], self.df[tf]["time"], wr
                    )

                fplt.plot(x, wr, legend=f"{tf} %R({period})", color=self.colors[tf], ax=ax)

        def plot_rsi(ax, period, color, levels: list):
            for i in range(len(levels)):
                fplt.add_line((0, levels[i]), (len(x), levels[i]), color='#fff', style='---', ax=ax)
            fplt.set_y_range(0, 100, ax=ax)
            fplt.plot(x, rsi(self.close, period), legend=f"RSI({period})", color=color, ax=ax)

        def plot_mfi(ax, period, color, levels: list):
            for i in range(len(levels)):
                fplt.add_line((0, levels[i]), (len(x), levels[i]), color='#fff', style='---', ax=ax)
            fplt.set_y_range(0, 100, ax=ax)
            fplt.plot(x, mfi(self.high, self.low, self.close, self.vol, period), legend=f"MFI({period})", color=color,
                      ax=ax)

        def plot_mml():
            for i in range(len(self.mml_d1)):
                if i != 4:
                    fplt.plot(x, self.mml_d1[i], color='blue', ax=ax1)
                else:
                    fplt.plot(x, self.mml_d1[i], color='red', ax=ax1)

        def plot_macd(ax):
            fplt.volume_ocv([x, self.open, self.close, self.macd_histogram], ax=ax, colorfunc=fplt.strength_colorfilter)
            fplt.plot(x, self.macd_line, ax=ax, legend='MACD')
            fplt.plot(x, self.macd_signal, ax=ax, legend='Signal')

        def plot_overlay_bars():
            overlay_plot = fplt.candlestick_ochl(
                [x, self.overlay_open, self.overlay_close, self.overlay_high, self.overlay_low]
            )
            overlay_plot.colors.update(dict(bull_body='#bfb', bull_shadow='#ada', bear_body='#fbc', bear_shadow='#dab'))

        def plot_slope(ax, data_arr, title, period, color):
            fplt.add_line((0, 0), (len(x), 0), color='#fff', style='---', ax=ax)
            fplt.plot(x, slope(data_arr, period), legend=title, color=color, ax=ax)

        def plot_ema(tf, period, color):
            ema_arr = ema(self.df[tf]["close"], period)

            if tf != self.strategy_dict["tfs"][0]:
                ema_arr = match_indexes(
                    self.df[self.strategy_dict["tfs"][0]]["time"], self.df[tf]["time"], ema_arr
                )

            fplt.plot(x, ema_arr, legend=f"EMA ({period})", color=color, ax=ax1)

        def plot_tema(tf, period, color):
            tema_arr = tema(self.df[tf]["close"], period)

            if tf != self.strategy_dict["tfs"][0]:
                tema_arr = match_indexes(
                    self.df[self.strategy_dict["tfs"][0]]["time"], self.df[tf]["time"], tema_arr
                )

            fplt.plot(x, tema_arr, legend=f"TEMA ({period})", color=color, ax=ax1)

        def plot_wpr_trend():
            fplt.plot(x, self.wpr_up_lines, legend="WPR Trend Up", width=marker_size, color="blue", ax=ax1)
            fplt.plot(x, self.wpr_down_lines, legend="WPR Trend Down", width=marker_size, color="red", ax=ax1)

        def plot_markers():
            fplt.plot(x, self.went_short_price, style='v', width=marker_size, legend="Went Short", color='purple',
                      ax=ax1)
            fplt.plot(x, self.went_long_price, style='^', width=marker_size, legend="Went Long", color='purple',
                      ax=ax1)
            fplt.plot(x, self.closed_long_profit, style='^', width=marker_size, legend="Closed Long Profit",
                      color='green', ax=ax1)
            fplt.plot(x, self.closed_long_loss, style='^', width=marker_size, legend="Closed Long Loss",
                      color='red', ax=ax1)
            fplt.plot(x, self.closed_short_profit, style='v', width=marker_size, legend="Closed Short Profit",
                      color='green', ax=ax1)
            fplt.plot(x, self.closed_short_loss, style='v', width=marker_size, legend="Closed Short Loss",
                      color='red', ax=ax1)

        [plotting_function(name) for name in self.tfs_dict.keys()]

        # plot_atr(ax1)
        # plot_rsi(ax2, 14, "yellow", [20, 80])
        # plot_wr(ax2, 21, [-20, -80], ["M15", "H1"])
        # plot_wr(ax2, 105, [-50], ["M15"])
        # plot_macd(ax2)
        # plot_mml()
        # plot_slope(ax4, self.)
        # plot_tema("D1", 50, "blue")
        # plot_supertrend()
        # plot_ema(self.strategy_dict["tfs"][-1], 10, "green")
        # plot_ema(self.strategy_dict["tfs"][-1], 50, "yellow")
        # plot_ema(self.strategy_dict["tfs"][-1], 200, "red")
        # plot_ema(self.strategy_dict['tfs'][0], 200, "red")
        # plot_ema(self.strategy_dict['tfs'][0], 50, "yellow")
        # plot_ema(self.strategy_dict['tfs'][0], 20, "white")
        # plot_ema("D1", 50, "yellow")
        # plot_ema("D1", 20, "white")
        # plot_ema("H4", 200, "red")
        # plot_ema("H1", 200, "red")
        # plot_wpr_trend()
        # plot_overlay_bars()
        plot_markers()

        # for n in range(self.high_peaks.shape[1]):
        #     fplt.plot(x, self.high_peaks[:, n], color='green', ax=ax1)
        #     fplt.plot(x, self.low_peaks[:, n], color='blue', ax=ax1)

        # fplt.plot(x, self.peaks_up, style="o", color='green', ax=ax1)
        # fplt.plot(x, self.peaks_down, style="o", color='red', ax=ax1)

        # fplt.plot(x, self.direction, color="white", ax=ax2)

        # fplt.plot(x, self.d1_supertrend, color='yellow', ax=ax1)
        # fplt.plot(x, self.green_supertrend, color='green', ax=ax1)
        # fplt.plot(x, self.superuptrend, color='green', ax=ax1)
        # fplt.plot(x, self.superdowntrend, color='red', ax=ax1)

        # fplt.plot(x, self.m5_supertrend, color='yellow', ax=ax1)
        # fplt.plot(x, self.m15_supertrend, color='red', ax=ax1)
        # fplt.plot(x, self.m30_supertrend, color='purple', ax=ax1)



        # fplt.plot(x, self.slow_superuptrend, color='green', ax=ax1)
        # fplt.plot(x, self.slow_superdowntrend, color='red', ax=ax1)
        # fplt.plot(x, self.med_superuptrend, color='green', ax=ax1)
        # fplt.plot(x, self.med_superdowntrend, color='red', ax=ax1)
        # fplt.plot(x, self.fast_superuptrend, color='green', ax=ax1)
        # fplt.plot(x, self.fast_superdowntrend, color='red', ax=ax1)

        # fplt.add_line((0, 20), (len(x), 20), color='#fff', style='---', ax=ax3)
        # fplt.add_line((0, 80), (len(x), 80), color='#fff', style='---', ax=ax3)
        # fplt.set_y_range(0, 100, ax=ax3)
        # fplt.set_y_range(-1, 1, ax=ax2)

        # fplt.plot(x, self.fast_stoch, legend="Fast Stochastic", color="purple", ax=ax3)
        # fplt.plot(x, self.slow_stoch, legend="Slow Stochastic", color="yellow", ax=ax3)
        # fplt.plot(x, self.fast_stoch_rsi, legend="Fast Stochastic RSI", color="blue", ax=ax3)
        # fplt.plot(x, self.slow_stoch_rsi, legend="Slow Stochastic RSI", color="red", ax=ax3)

        # fplt.plot(x, self.dema, legend="DEMA 200", color="blue", ax=ax1)
        # fplt.plot(x, self.d1_hma, legend="HMA D1", color="purple", ax=ax1)
        # fplt.plot(x, self.h1_hma, legend="HMA H1", color="green", ax=ax1)
        # fplt.plot(x, self.hma, legend="HMA", color="purple", ax=ax1)

        # fplt.volume_ocv([x, self.open, self.close, self.tsv_histogram], ax=ax2, colorfunc=fplt.strength_colorfilter)
        # fplt.plot(x, self.tsv_ma, ax=ax2, legend='TSV Signal')

        # fplt.plot(x, self.trend, legend="M1 Trend", color="white", ax=ax2)
        # fplt.plot(x, self.concentration, legend="M1 Concentration", color="red", ax=ax2)
        # fplt.plot(x, self.m15_trend, legend="M15 Trend", color="red", ax=ax2)
        # fplt.plot(x, self.tma_direction, legend="Direction", color="white", ax=ax2)
        # fplt.plot(x, self.classic_trend, legend="Classic Trend", color="red", ax=ax2)
        # fplt.plot(x, self.strength, legend="Strength M1", color="white", ax=ax4)
        # fplt.plot(x, self.h1_trend, legend="H1 Trend", color="red", ax=ax3)
        # fplt.plot(x, self.strength, legend="Strength M1", color="white", ax=ax2)
        # fplt.plot(x, self.h1_strength, legend="Strength H1", color="blue", ax=ax2)
        # fplt.plot(x, self.d1_strength, legend="Strength D1", color="white", ax=ax2)
        # fplt.plot(x, self.h4_strength, legend="Strength H4", color="green", ax=ax2)
        # fplt.add_line((0, 0), (len(x), 0), color='#fff', style='---', ax=ax4)

        # fplt.plot(x, self.trading_range, legend="Trading Range", color="white", ax=ax3)

        # fplt.plot(x, self.m15_strength, legend="Strength M15", color="red", ax=ax2)
        # fplt.plot(x, self.tma_slope_upper, legend="TMA Slope Upper", color="yellow", ax=ax5)
        # fplt.plot(x, self.tma_slope_lower, legend="TMA Slope Lower", color="purple", ax=ax5)
        # fplt.plot(x, self.ema_20_slope, legend="M15 EMA 20 Slope(1)", color="yellow", ax=ax2)
        # fplt.add_line((0, 0), (len(x), 0), color='#fff', style='---', ax=ax2)
        # fplt.plot(x, self.m1_tma_width, legend="M1 TMA Width",
        #           color="white", ax=ax5)

        # fplt.plot(x, self.qqe[0], legend="Smoothed RSI", color="blue", ax=ax2)
        # fplt.plot(x, self.qqe[1], legend="Slow Trailing Line", style="---", color="red", ax=ax2)

        # fplt.volume_ocv([x, self.open, self.close, self.vol], ax=ax1.overlay())
        # overlay_plot = fplt.candlestick_ochl(
        #     [x, self.open_ha, self.close_ha, self.high_ha, self.low_ha]
        # )
        # overlay_plot.colors.update(dict(bull_body='#bfb', bull_shadow='#ada', bear_body='#fbc', bear_shadow='#dab'))
        fplt.candlestick_ochl([x, self.open, self.close, self.high, self.low], ax=ax1)
        # fplt.candlestick_ochl([x, self.open_ha, self.close_ha, self.high_ha, self.low_ha], ax=ax1)

        fplt.show()
