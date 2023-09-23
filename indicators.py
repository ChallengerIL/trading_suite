import numpy as np
from numba import njit, prange, jit
import math
from scipy.signal import find_peaks
import datetime as dt
import finplot as fplt
# import functools

# # Nested attributes
# def rsetattr(obj, attr, val):
#     pre, _, post = attr.rpartition('.')
#     return setattr(rgetattr(obj, pre) if pre else obj, post, val)
#
#
# def rgetattr(obj, attr, *args):
#     def _getattr(obj, attr):
#         return getattr(obj, attr, *args)
#     return functools.reduce(_getattr, [obj] + attr.split('.'))


class Indicator(object):

    def __init__(self, name, df, strategy, tfs):
        self.indicators_dict = {
            "hma": hma, "tma": tma, "tsv": tsv, "ichimoku_cloud": ichimoku_cloud,
            "wr": get_wr, "supertrend": supertrend, "mml": mml_calculator, "ema": ema,
            "dema": dema, "tema": tema, "wma": wma, "rsi": rsi, "mfi": mfi,
        }

        self.name = name
        self.df = df
        self.strategy = strategy
        self.all_tfs = tfs

        self.tfs = sorted(list(set(self.strategy['indicators'][self.name].keys())),
                          key=self.strategy['tfs'].index, reverse=True)

        self.calculate()

    def calculate(self):
        for tf in self.tfs:
            indicators_input = {
                "ichimoku_cloud": {"h": self.df[tf]["high"], "l": self.df[tf]["low"]},
                "hma": {"c": self.df[tf]["close"]},
                "tma": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"]},
                "tsv": {"c": self.df[tf]["close"], "v": self.df[tf]["vol"]},
                "wr": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"]},
                "supertrend": {"h": self.df[tf]["high"], "l": self.df[tf]["low"], "c": self.df[tf]["close"]},
                "mml": {"h": self.df[tf]["high"], "l": self.df[tf]["low"]},
                "ema": {"c": self.df[tf]["close"]},
                "dema": {"c": self.df[tf]["close"]},
                "tema": {"c": self.df[tf]["close"]},
                "wma": {"c": self.df[tf]["close"]},
                "rsi": {"c": self.df[tf]["close"]},
                "mfi": {"h": self.df[tf]["high"], "l": self.df[tf]["low"],
                        "c": self.df[tf]["close"], "v": self.df[tf]["vol"]},
            }

            params = self.strategy['indicators'][self.name][tf]

            loops_num = 1

            if 'p' in params.keys():
                loops_num = len(params['p'])

            for i in range(loops_num):
                params_to_add_dict = dict()

                for params_tuple in params.items():
                    if params_tuple[0] != "plotting":
                        params_to_add_dict[params_tuple[0]] = params_tuple[1][i]

                indicators_input[self.name].update(params_to_add_dict)

                data = self.indicators_dict[self.name](**indicators_input[self.name])

                result = list()

                if tf != self.all_tfs[0]:
                    # Check if we are dealing with a nested array/list
                    if any(isinstance(i, (np.ndarray, list)) for i in data):
                        for idx in range(len(data)):
                            result.append(match_indexes(
                                self.df[self.all_tfs[0]]["time"], self.df[tf]["time"], data[idx]))
                    else:
                        result = match_indexes(
                            self.df[self.all_tfs[0]]["time"], self.df[tf]["time"], data
                        )
                else:
                    result = data

                new_name = f"{tf.lower()}"

                if 'p' in params.keys():
                    new_name += f"_{params['p'][i]}"

                setattr(self, new_name, result)

    def plot(self, index, ax_list: list):
        for tf in self.tfs:
            params = self.strategy['indicators'][self.name][tf]

            loops_num = 1

            if 'p' in params.keys():
                loops_num = len(params['p'])

            for idx in range(loops_num):

                name = f"{tf.lower()}"

                if 'p' in params.keys():
                    name += f"_{params['p'][idx]}"

                data = getattr(self, name)

                ax = ax_list[0]

                if 'ax' in params['plotting'].keys():
                    ax = ax_list[params['plotting']["ax"][idx]]

                if 'levels' in params['plotting'].keys():
                    fplt.set_y_range(params['plotting']['ymin'], params['plotting']['ymax'], ax=ax)

                    for level in params['plotting']["levels"][idx]:
                        fplt.add_line((0, level), (len(index), level), color='#fff', style='---', ax=ax)

                if any(isinstance(i, (np.ndarray, list)) for i in data):
                    if self.name == 'tsv':
                        fplt.volume_ocv([index, self.df[tf]["open"],
                                        self.df[tf]["close"], data[0]], ax=ax,
                                        colorfunc=fplt.strength_colorfilter)
                        fplt.plot(index, data[1], ax=ax, legend=f'TSV Signal {tf}')
                    else:
                        plotted = [fplt.plot(index, d, color=params['plotting']['color'][idx], ax=ax) for d in data]

                        if self.name == 'ichimoku_cloud':
                            fplt.fill_between(plotted[0], plotted[1], color=params['plotting']['color'][idx])
                else:
                    fplt.plot(index, data, legend=f"{self.name.upper()} {tf} ({params['p'][idx]})", color=params['plotting']['color'][idx], ax=ax)


@njit
def tsv(c, v, p=13, ma_p=7):
    # Requires confirmation

    first_run = np.zeros_like(c)
    histogram = np.zeros_like(c)

    for i in range(1, len(c)):
        if not c[i] == c[i-1]:
            first_run[i] = v[i] * (c[i] - c[i - 1])

    for i in range(p, len(c)):
        histogram[i] = np.sum(first_run[i-p:i])

    t_ma = sma(histogram, ma_p)

    return histogram, t_ma


@njit
def hma(c, p=55):
    # Requires confirmation

    return wma(2 * wma(c, p // 2) - wma(c, p), round(math.sqrt(p)))


def peaks(h, l):
    peaks_h, _ = find_peaks(h)
    peaks_l, _ = find_peaks(-l)

    return peaks_h, peaks_l


# @njit
def trend_lines(h, l, c, plotting=True):
    # price moves in one direction, start looking for a line once there is a bar in the opposite direction
    # connect the lines when there is no abstraction between the start line and the current min/max

    cols = 20

    # peaks_up = np.empty(len(h))
    # peaks_down = np.empty(len(h))
    # peaks_up[:] = np.nan
    # peaks_down[:] = np.nan
    #
    # for i in peaks_h:
    #     peaks_up[i] = h[i]
    #
    # for i in peaks_l:
    #     peaks_down[i] = l[i]

    # for checking how close price is to the line use np.isclose

    trend_direction = np.zeros_like(h)

    up_trend = np.empty((len(h), cols))
    down_trend = np.empty((len(l), cols))
    up_trend[:] = np.nan
    down_trend[:] = np.nan

    up_angle = np.empty((len(h), cols))
    up_angle[:] = np.nan
    down_angle = np.empty((len(h), cols))
    down_angle[:] = np.nan

    # Get rid of looking into the future.
    # Add lines to the past for plotting (don't use for backtesting)

    #  np.isnan(up_angle[i-1, col]
    # np.any(up_angle[i - 1, :]

    for col in range(cols):
        for i in range(2, len(h)):
            if np.isnan(up_angle[i-1, col]):
                if l[i] > l[i - 1]:
                    if np.any(up_angle[i-1, :]) != l[i] - l[i-1]:
                        if plotting:
                            up_trend[i-1, col] = l[i-1]

                        up_angle[i, col] = l[i] - l[i-1]
                        up_trend[i, col] = l[i]

            if not np.isnan(up_angle[i-1, col]):
                up_trend[i, col] = up_trend[i-1, col] + up_angle[i-1, col]
                if not c[i] < up_trend[i-1, col] + up_angle[i-1, col]:
                    up_angle[i, col] = up_angle[i-1, col]

    print(up_trend[-10:])


    # # Iterate through matrix columns
    # for n in range(cols):
    #     # Iterate through matrix rows
    #     for i in range(1, len(h)):
    #         # If current low is higher than the previous one
    #         if l[i] > l[i-1]:
    #             # If there is no other lines in the current row at this index
    #             if np.isnan(up_angle[i-1, n]):
    #                 #
    #                 if up_angle[i-1, :].all() != l[i] - l[i-1]:
    #                     if plotting:
    #                         up_trend[i-1, n] = l[i-1]
    #                         up_angle[i-1:i, n] = l[i] - l[i-1]
    #
    #                     up_trend[i, n] = l[i]
    #                     up_angle[i, n] = l[i] - l[i-1]
    #
    #         if not np.isnan(up_angle[i-1, n]):
    #             up_trend[i, n] = up_trend[i-1, n] + up_angle[i-1, n]
    #             if not c[i] < up_trend[i, n]:
    #                 up_angle[i, n] = up_angle[i-1, n]
    #
    #         if h[i] < h[i-1]:
    #             if np.isnan(down_angle[i-1, n]):
    #                 if down_angle[i-1, :].all() != h[i] - h[i-1]:
    #                     if plotting:
    #                         down_trend[i-1, n] = h[i-1]
    #                         down_angle[i-1:i, n] = h[i] - h[i-1]
    #
    #                     down_trend[i, n] = h[i]
    #                     down_angle[i, n] = h[i] - h[i-1]
    #
    #         if not np.isnan(down_angle[i-1, n]):
    #             down_trend[i, n] = down_trend[i-1, n] + down_angle[i-1, n]
    #             if not c[i] > down_trend[i, n]:
    #                 down_angle[i, n] = down_angle[i-1, n]

    # print(up_trend[-10:])

    # [1.27432 1.33405 1.27432 1.33405 1.27432 1.33405 1.27432 1.33405 1.27432
    #   1.33405 1.27432 1.33405 1.27432 1.33405 1.27432 1.33405 1.27432 1.33405
    #   1.27432 1.33405]
    #  [1.27446 1.33645 1.27446 1.33645 1.27446 1.33645 1.27446 1.33645 1.27446
    #   1.33645 1.27446 1.33645 1.27446 1.33645 1.27446 1.33645 1.27446 1.33645
    #   1.27446 1.33645]
    #  [1.2746  1.33885 1.2746  1.33885 1.2746  1.33885 1.2746  1.33885 1.2746
    #   1.33885 1.2746  1.33885 1.2746  1.33885 1.2746  1.33885 1.2746  1.33885
    #   1.2746  1.33885]]

    # Find all rows [0] and columns [1] with NaNs
    # nans = np.where(np.isnan(up_trend))

    return up_trend, down_trend


@njit
def inventory_retracement_bar(o, h, l, c, percentage=45):
    result = np.zeros_like(o)

    candle_size = h - l
    high_wick = 100 / candle_size * np.where(c > o, h - c, h - o)
    low_wick = 100 / candle_size * np.where(c > o, o - l, c - l)

    for i in range(len(o)):
        if high_wick[i] >= percentage:
            result[i] = 1
        if low_wick[i] >= percentage:
            result[i] = -1

    return result


@njit(parallel=True)
def ema(c, p, alpha=False):
    if alpha:
        alpha = 1 / p
    else:
        alpha = 2 / (p + 1)

    start = 0

    if np.isnan(c).any():
        start = np.where(np.isnan(c))[0][-1] + 1

    exp_weights = np.zeros(len(c))
    exp_weights[start + p - 1] = np.mean(c[start:start + p])

    for i in range(start + p, len(exp_weights) + 1):
        exp_weights[i] = exp_weights[i - 1] * (1 - alpha) + (alpha * (c[i]))
    exp_weights[:start + p - 1] = np.nan

    return exp_weights


@njit
def tema(c, p):
    ema_1 = ema(c, p)
    ema_2 = ema(ema_1, p)

    return (3 * ema_1) - (3 * ema_2) + ema(ema_2, p)


@njit
def dema(c, p):
    ema_arr = ema(c, p)

    return (2 * ema_arr) - ema(ema_arr, p)


# @njit
def qqe(rsi_arr):
    rsi_period = 14
    rsi_smoothing_factor = 5
    qqe_multiplier = 4.236
    wilders_period = rsi_period * 2 - 1

    qqe_line = ema(rsi_arr, rsi_smoothing_factor)
    rsi_atr = np.abs(shift(qqe_line, 1) - qqe_line)
    dar = ema(ema(rsi_atr, wilders_period), wilders_period) * qqe_multiplier

    new_short_band = qqe_line + dar
    new_long_band = qqe_line - dar

    long_band = np.zeros_like(qqe_line)
    long_band = np.where(
        shift(qqe_line, 1) > shift(long_band, 1) and qqe_line > shift(long_band, 1),
        max(shift(long_band, 1), new_long_band), new_long_band)

    print(long_band)
    quit()

    # df['shortband'] = 0.0
    # df['shortband'] = np.where(
    #     df['RSI_ma'].shift(1) < df['shortband'].shift(1) and df['RSI_ma'] < df['shortband'].shift(1),
    #     max(df['shortband'].shift(1), df['newshortband']), df['newshortband'])
    #
    # df['trend'] = np.where(df['RSI_ma'] > df['dar'].shift(1), 1, -1)
    # df['FastAtrRsiTL'] = np.where(df['trend'] == 1, df['longband'], df['shortband'])

    return qqe_line


@njit(parallel=True)
def heikin_ashi(o, h, l, c):
    open_ha = np.empty(len(o), dtype=o.dtype)
    high_ha = np.empty(len(o), dtype=o.dtype)
    low_ha = np.empty(len(o), dtype=o.dtype)
    open_ha[0] = o[0]

    close_ha = (o + h + l + c) / 4

    for i in range(1, len(o)):
        open_ha[i] = (open_ha[i-1] + close_ha[i-1]) / 2

    for i in range(len(o)):
        high_ha[i] = max(o[i], h[i], l[i], c[i])
        low_ha[i] = min(o[i], h[i], l[i], c[i])

    return open_ha, high_ha, low_ha, close_ha


@njit
def wpr_trend(wpr, h, l, c, strength_period, multiplier, strength_add_number=0.5):
    arr_len = len(c)
    strength = np.zeros_like(c)
    direction = np.zeros_like(c)
    trend = np.zeros_like(c, dtype=np.int32)
    classic_trend = np.zeros_like(c, dtype=np.int32)
    concentration = np.zeros_like(c, dtype=np.int32)
    up_lines = np.empty(arr_len)
    up_lines[:] = np.NaN
    down_lines = np.empty(arr_len)
    down_lines[:] = np.NaN
    up_picks = list()
    down_picks = list()
    current_pick = 0
    flat_start = 0
    reversal = 0

    for i in range(2, len(c)):
        # If the current direction is up
        if direction[i-2] == 1:
            # Check if the price has crossed -20 level from up down, which means a possible reversal
            if wpr[i-2] >= -20 > wpr[i-1]:
                strength_holder = (np.sum(strength[i-strength_period:i]) / strength_period) + strength_add_number
                reversal = -1
                direction[i-1] = 1
                if h[i-1] > current_pick:
                    strength_holder += (h[i-1] - current_pick) / multiplier
                    current_pick = h[i-1]
                up_lines[i-1] = current_pick
                strength[i-1] = strength_holder
            # If the current trend is steady in the upward direction
            elif wpr[i-1] > -80:
                strength_holder = (np.sum(strength[i-strength_period:i]) / strength_period) + strength_add_number
                if wpr[i-1] >= -20:
                    strength_holder += strength_add_number
                if reversal == -1 and wpr[i-1] >= -20:
                    reversal = 0
                if h[i-1] > current_pick:
                    strength_holder += (h[i-1] - current_pick) / multiplier
                    current_pick = h[i - 1]
                up_lines[i-1] = current_pick
                direction[i-1] = 1
                strength[i-1] = strength_holder
            # The trend has reversed
            else:
                direction[i-1] = -1
                strength[i-1] = (np.sum(strength[i-strength_period:i]) / strength_period) - strength_add_number
                reversal = 0
                up_picks.append(current_pick)
                current_pick = l[i-1]
                down_lines[i-1] = current_pick

        # If the current direction is down
        elif direction[i-2] == -1:
            # Check if the price has crossed -80 level from down up, which means a possible reversal
            if wpr[i-2] <= -80 < wpr[i-1]:
                strength_holder = (np.sum(strength[i-strength_period:i]) / strength_period) - strength_add_number
                reversal = 1
                direction[i-1] = -1
                if l[i-1] < current_pick:
                    strength_holder -= (current_pick - l[i-1]) / multiplier
                    current_pick = l[i-1]
                down_lines[i-1] = current_pick
                strength[i-1] = strength_holder
            # If the current trend is steady in the downward direction
            elif wpr[i-1] < -20:
                strength_holder = (np.sum(strength[i-strength_period:i]) / strength_period) - strength_add_number
                if wpr[i-1] <= -80:
                    strength_holder -= strength_add_number
                if reversal == 1 and wpr[i-1] <= -80:
                    reversal = 0
                if l[i-1] < current_pick:
                    strength_holder -= (current_pick - l[i-1]) / multiplier
                    current_pick = l[i-1]
                down_lines[i-1] = current_pick
                direction[i-1] = -1
                strength[i-1] = strength_holder
            # The trend has reversed
            else:
                direction[i-1] = 1
                strength[i-1] = (np.sum(strength[i-strength_period:i]) / strength_period) + strength_add_number
                reversal = 0
                down_picks.append(current_pick)
                current_pick = h[i-1]
                up_lines[i-1] = current_pick

        elif direction[i-1] == 0:
            if wpr[i-1] >= -20 > wpr[i-2]:
                direction[i-1] = 1
                strength[i-1] = strength_add_number
                current_pick = h[i-1]
                up_lines[i-1] = current_pick
            elif wpr[i-1] <= -80 < wpr[i-2]:
                direction[i-1] = -1
                strength[i-1] = -strength_add_number
                current_pick = l[i-1]
                down_lines[i-1] = current_pick

        # Trend and Flat
        if len(down_picks) > 2 and len(up_picks) > 2:

            # Classic Trend
            if up_picks[-1] > up_picks[-2] and down_picks[-1] > down_picks[-2]:
                classic_trend[i-1] = 1
                if flat_start != 0:
                    flat_start = 0
            elif up_picks[-1] < up_picks[-2] and down_picks[-1] < down_picks[-2]:
                classic_trend[i-1] = -1
                if flat_start != 0:
                    flat_start = 0
            else:
                if classic_trend[i-2] == 1 and not c[i-1] < down_picks[-1]:
                    classic_trend[i-1] = classic_trend[i-2]
                elif classic_trend[i-2] == -1 and not c[i-1] > up_picks[-1]:
                    classic_trend[i-1] = classic_trend[i-2]

            # Regular Trend
            if flat_start == 0:
                if h[i - 1] > up_picks[-1] and down_picks[-1] > down_picks[-2]:
                    trend[i - 1] = 1
                elif l[i - 1] < down_picks[-1] and up_picks[-1] < up_picks[-2]:
                    trend[i - 1] = -1
                else:
                    trend[i - 1] = trend[i - 2]

                    if trend[i-2] == 0:
                        if down_picks[-1] > down_picks[-2]:
                            if h[i - 1] > up_picks[-2]:
                                trend[i - 1] = 1
                        elif up_picks[-1] < up_picks[-2]:
                            if l[i - 1] < down_picks[-2]:
                                trend[i - 1] = -1

                    # if trend[i-2] == 1:
                    #     if c[i-1] < down_picks[-1]:
                    #         trend[i-1] = 0
                    #     else:
                    #         trend[i - 1] = trend[i - 2]
                    # elif trend[i-2] == -1:
                    #     if c[i-1] > up_picks[-1]:
                    #         trend[i-1] = 0
                    #     else:
                    #         trend[i - 1] = trend[i - 2]
                    # else:
                    #     if down_picks[-1] > down_picks[-2]:
                    #         if c[i-1] > up_picks[-2]:
                    #             trend[i-1] = 1
                    #     elif up_picks[-1] < up_picks[-2]:
                    #         if c[i-1] < down_picks[-2]:
                    #             trend[i-1] = -1

            # # Regular Trend
            # if flat_start == 0:
            #     if c[i-1] > up_picks[-1] and down_picks[-1] > down_picks[-2]:
            #         trend[i-1] = 1
            #     elif c[i-1] < down_picks[-1] and up_picks[-1] < up_picks[-2]:
            #         trend[i-1] = -1
            #     else:
            #         trend[i-1] = trend[i-2]

            # Concentration
            if up_picks[-1] < up_picks[-2] and down_picks[-1] > down_picks[-2]:
                if down_picks[-2] > down_picks[-3]:
                    if h[i-1] > up_picks[-1] > up_lines[i-2]:
                        concentration[i-1] = 1
                if up_picks[-2] < up_picks[-3]:
                    if l[i-1] < down_picks[-1] < down_lines[i-2]:
                        concentration[i-1] = -1

            # Flat
            if classic_trend[i - 1] == 0:
                if up_picks[-1] < up_picks[-3] and up_picks[-2] < up_picks[-3] and \
                        down_picks[-1] > down_picks[-3] and down_picks[-2] > down_picks[-3]:
                    flat_start = i - 1
                if flat_start != 0:
                    if h[i - 1] > h[flat_start] or l[i - 1] < l[flat_start]:
                        flat_start = 0
                        if h[i - 1] > h[flat_start]:
                            trend[i - 1] = 1
                        else:
                            trend[i - 1] = -1
                    else:
                        trend[i - 1] = 0

            # # Flat
            # if classic_trend[i-1] == 0:
            #     if up_picks[-1] < up_picks[-3] and up_picks[-2] < up_picks[-3] and \
            #             down_picks[-1] > down_picks[-3] and down_picks[-2] > down_picks[-3]:
            #         flat_start = i-1
            #     if flat_start != 0:
            #         if c[i-1] > c[flat_start] or c[i-1] < c[flat_start]:
            #             flat_start = 0
            #             if c[i-1] > c[flat_start]:
            #                 trend[i-1] = 1
            #             else:
            #                 trend[i-1] = -1
            #         else:
            #             trend[i-1] = 0

    return trend, classic_trend, concentration, ema(strength, strength_period), up_lines, down_lines


def stochastic(arr, p=14, sma_p=3):
    rolling_max_arr = rolling_max(arr, p)
    rolling_min_arr = rolling_min(arr, p)

    fast_stoch = sma(((arr - rolling_min_arr) / (rolling_max_arr - rolling_min_arr)) * 100, sma_p)
    slow_stoch = sma(fast_stoch, sma_p)

    return fast_stoch, slow_stoch


@njit
def supertrend(h, l, c, p=12, multiplier=3):
    atr_arr = atr(h, l, c, p)
    basic_upper_band = (h + l) / 2 + atr_arr * multiplier
    basic_lower_band = (h + l) / 2 - atr_arr * multiplier

    final_upper_band = np.zeros_like(h)
    final_lower_band = np.zeros_like(h)
    result = np.zeros_like(h)

    superuptrend = np.empty(len(h))
    superdowntrend = np.empty(len(h))

    superuptrend[:] = np.nan
    superdowntrend[:] = np.nan

    for i in range(p, len(h)):
        if basic_upper_band[i] < final_upper_band[i-1] or c[i-1] > final_upper_band[i-1]:
            final_upper_band[i] = basic_upper_band[i]
        else:
            final_upper_band[i] = final_upper_band[i-1]

        if basic_lower_band[i] > final_lower_band[i - 1] or c[i-1] < final_lower_band[i-1]:
            final_lower_band[i] = basic_lower_band[i]
        else:
            final_lower_band[i] = final_lower_band[i-1]

    for i in range(p, len(h)):
        if result[i-1] == final_upper_band[i-1] and c[i] <= final_upper_band[i]:
            result[i] = final_upper_band[i]
            superdowntrend[i] = final_upper_band[i]
        else:
            if result[i-1] == final_upper_band[i-1] and c[i] > final_upper_band[i]:
                result[i] = final_lower_band[i]
                superuptrend[i] = final_lower_band[i]
            else:
                if result[i-1] == final_lower_band[i-1] and c[i] >= final_lower_band[i]:
                    result[i] = final_lower_band[i]
                    superuptrend[i] = final_lower_band[i]
                else:
                    if result[i-1] == final_lower_band[i-1] and c[i] < final_lower_band[i]:
                        result[i] = final_upper_band[i]
                        superdowntrend[i] = final_upper_band[i]

    result[:p] = np.nan

    return result, superuptrend, superdowntrend


@njit(parallel=True)
def macd(close, fast_ema=12, slow_ema=26, signal_ema=9):
    macd_line = ema(close, fast_ema) - ema(close, slow_ema)
    macd_line_temp = macd_line[~np.isnan(macd_line)]
    signal_line_holder = ema(macd_line_temp, signal_ema)
    signal_line = np.empty((len(close)))
    signal_line[:] = np.NaN
    signal_line[len(close)-len(signal_line_holder):] = signal_line_holder
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


@jit(forceobj=True)
def index_to_datetime(index):
    return np.array([dt.datetime.utcfromtimestamp(time) for time in index], dtype=object)


@njit
def numpy_fill(arr):
    out = arr.copy()

    for row_idx in range(out.shape[0]):
        if np.isnan(out[row_idx]):
            out[row_idx] = out[row_idx - 1]

    return out


@njit
def match_indexes(goal_index, source_index, arr, shift_to_match=True, arr_type=np.float64):
    idx_arr = np.empty(len(goal_index))
    idx_arr[:] = np.nan
    step = int((source_index[1] - source_index[0]) / (goal_index[1] - goal_index[0]))

    for i in prange(len(goal_index)):
        for n in prange(i // step, len(source_index)):
            if source_index[n] == goal_index[i]:
                idx_arr[i] = arr[n]
                break

    result = numpy_fill(idx_arr)
    result = result.astype(arr_type)
    if shift_to_match:
        result = shift(result, step)

    return result


@njit
def sr_calculator(high):
    result = 0
    # if 250000 > max_number > 25000:
    #     result = 100000
    # elif 25000 > max_number > 2500:
    #     result = 10000
    # elif 2500 > max_number > 250:
    #     result = 1000
    if 1.5625 > high > 0.390625:
        result = 1.5625
    elif 250 > high > 25:
        result = 100
    elif 25 > high > 12.5:
        result = 12.5
    elif 12.5 > high > 6.25:
        result = 12.5
    elif 6.25 > high > 3.125:
        result = 6.25
    elif 3.125 > high > 1.5625:
        result = 3.125
    elif 0.390625 > high > 0:
        result = 0.1953125

    return result


@njit
def allowed_squares(range_mmi):
    if 5 <= range_mmi < 999:
        squares = np.array([[0, 8], [4, 4]])
    elif 3 <= range_mmi < 5:
        squares = np.array([[0, 4], [2, 6], [4, 8]])
    elif 1 <= range_mmi < 3:
        squares = np.array([[0, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 8]])

    return squares


@njit
def mml_calculator(h, l, window=64):
    # Setting Period:
    # - For scalping (TF 5m/1m): 32
    # - For Intraday / Swing (15m TF-daily): 64(default)
    # - For long-term (TF-yearly weekly): 128

    max_number = rolling_max(h, window)
    max_number = max_number[window - 1:]
    min_number = rolling_min(l, window)
    min_number = min_number[window - 1:]
    zipped_prices = zip(max_number, min_number)
    price_range = [x - n for x, n in zipped_prices]

    sr = [sr_calculator(x) for x in max_number]

    octave_count = list()
    comparing_num = 1.25
    mmi = list()

    major_mmi = [x / 8 for x in sr]
    minor_mmi = [x / 8 / 8 for x in sr]
    baby_mmi = [x / 8 / 8 / 8 for x in sr]
    micro_mmi = [x / 8 / 8 / 8 / 8 for x in sr]
    extra_micro_mmi = [x / 8 / 8 / 8 / 8 / 8 for x in sr]

    for i, p in enumerate(price_range):
        if p / major_mmi[i] > comparing_num:
            mmi.append(major_mmi[i])
            octave_count.append(1)
        elif p / minor_mmi[i] > comparing_num:
            mmi.append(minor_mmi[i])
            octave_count.append(2)
        elif p / baby_mmi[i] > comparing_num:
            mmi.append(baby_mmi[i])
            octave_count.append(3)
        elif p / micro_mmi[i] > comparing_num:
            mmi.append(micro_mmi[i])
            octave_count.append(4)
        elif p / extra_micro_mmi[i] > comparing_num:
            mmi.append(extra_micro_mmi[i])
            octave_count.append(5)

    range_mmi_zip = zip(price_range, mmi)

    range_mmi = [math.floor(p / m) for p, m in range_mmi_zip]
    squares = [allowed_squares(x) for x in range_mmi]
    base = list()
    base_holder = list()

    for i, o in enumerate(octave_count):
        for x in range(o):
            if x == 0:
                base.append(min_number[i] - 0)
                holder = math.floor(base[i] / major_mmi[i])
                base[i] = 0 + (holder * major_mmi[i])
            elif x == 1:
                base_holder.append(base[i])
                base[i] = min_number[i] - base[i]
                holder = math.floor(base[i] / minor_mmi[i])
                base[i] = base_holder[i] + (holder * minor_mmi[i])
            elif x == 2:
                base_holder[i] = base[i]
                base[i] = min_number[i] - base[i]
                holder = math.floor(base[i] / baby_mmi[i])
                base[i] = base_holder[i] + (holder * baby_mmi[i])
            elif x == 3:
                base_holder[i] = base[i]
                base[i] = min_number[i] - base[i]
                holder = math.floor(base[i] / micro_mmi[i])
                base[i] = base_holder[i] + (holder * micro_mmi[i])
            elif x == 4:
                base_holder[i] = base[i]
                base[i] = min_number[i] - base[i]
                holder = math.floor(base[i] / extra_micro_mmi[i])
                base[i] = base_holder[i] + (holder * extra_micro_mmi[i])

    sq_list = list()
    base_mmi_zip = zip(base, mmi)
    bottom_mml = [b - m for b, m in base_mmi_zip]

    for i, s in enumerate(squares):
        holder = list()
        for sq in s:
            test_bottom = bottom_mml[i] + sq[0] * mmi[i]
            test_top = bottom_mml[i] + sq[1] * mmi[i]
            holder.append([test_bottom, test_top])
        sq_list.append(holder)

    error_list = list()

    for i, s in enumerate(sq_list):
        holder = list()
        for sq in s:
            error_func = abs(max_number[i] - sq[1]) + abs(min_number[i] - sq[0])
            holder.append(error_func)
        error_list.append(holder)

    lowest_error = list()
    for e in error_list:
        lowest_error.append(e.index(min(e)))

    top = list()
    bottom = list()
    for i, sq in enumerate(sq_list):
        top.append(sq[lowest_error[i]][1])
        bottom.append(sq[lowest_error[i]][0])

    top_bottom_zip = zip(top, bottom)
    mml_step = [(t - b) / 8 for t, b in top_bottom_zip]

    current_mml = [round(b, 5) for b in bottom]

    list_0 = list()
    list_1 = list()
    list_2 = list()
    list_3 = list()
    list_4 = list()
    list_5 = list()
    list_6 = list()
    list_7 = list()
    list_8 = list()

    for i, c in enumerate(current_mml):
        current_list = list()
        current_list.append(c)

        for n in range(8):
            c += mml_step[i]
            current_list.append(round(c, 5))

        list_0.append(current_list[0])
        list_1.append(current_list[1])
        list_2.append(current_list[2])
        list_3.append(current_list[3])
        list_4.append(current_list[4])
        list_5.append(current_list[5])
        list_6.append(current_list[6])
        list_7.append(current_list[7])
        list_8.append(current_list[8])

    nans = np.empty(window - 1)
    nans[:] = np.nan
    list_0 = np.append(nans, list_0)
    list_1 = np.append(nans, list_1)
    list_2 = np.append(nans, list_2)
    list_3 = np.append(nans, list_3)
    list_4 = np.append(nans, list_4)
    list_5 = np.append(nans, list_5)
    list_6 = np.append(nans, list_6)
    list_7 = np.append(nans, list_7)
    list_8 = np.append(nans, list_8)

    return list_0, list_1, list_2, list_3, list_4, list_5, list_6, list_7, list_8


@njit(parallel=True)
def ichimoku_cloud(h, l):
    high_9 = rolling_max(h, 9)
    low_9 = rolling_min(l, 9)
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = rolling_max(h, 26)
    low_26 = rolling_min(l, 26)
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_a = shift((tenkan_sen + kijun_sen) / 2, 26)

    high_52 = rolling_max(h, 52)
    low_52 = rolling_min(l, 52)
    senkou_span_b = shift((high_52 + low_52) / 2, 26)

    return senkou_span_a, senkou_span_b


@njit(parallel=True)
def tma(h, l, c, p=20, atr_p=100, multiplier=2):
    atr_value = atr(h, l, c, atr_p)
    middle_line = wma(c, p)
    upper_line = middle_line + atr_value * multiplier
    lower_line = middle_line - atr_value * multiplier

    return lower_line, middle_line, upper_line


@njit(parallel=True)
def wma(c, p):
    wma_range = len(c) - p + 1
    wmas = np.zeros(wma_range)
    k = (p * (p + 1)) / 2.0

    for idx in range(wma_range):
        for period_num in range(p):
            weight = period_num + 1
            wmas[idx] += c[idx + period_num] * weight
        wmas[idx] /= k

    nans = np.empty(p - 1)
    nans[:] = np.nan
    result = np.append(nans, wmas)

    return result


@njit
def wills_fractals(h, l):
    result = np.zeros_like(h)

    for i in range(2, len(h) - 2):
        if h[i] > h[i - 1]:
            if h[i] > h[i - 2]:
                if h[i] > h[i + 1]:
                    if h[i] > h[i + 2]:
                        result[i] = -1
        elif l[i] < l[i - 1]:
            if l[i] < l[i - 2]:
                if l[i] < l[i + 1]:
                    if l[i] < l[i + 2]:
                        result[i] = 1

    return result


@njit(parallel=True)
def get_wr(h, l, c, p=21):
    high = rolling_max(h, p)
    low = rolling_min(l, p)

    return -100 * ((high - c) / (high - low))


@njit
def atr(h, l, c, p):
    tr = np.zeros_like(h)
    tr[0] = h[0] - l[0]

    for i in range(1, len(h)):
        hl = h[i] - l[i]
        hpc = abs(h[i] - c[i - 1])
        lpc = abs(l[i] - c[i - 1])
        tr[i] = max(max(hl, hpc), lpc)

    return sma(tr, p)


@njit(parallel=True)
def slope(value, p):
    return value - shift(value, p)


@njit(parallel=True)
def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


@njit(parallel=True)
def rolling_max(data, n):
    ar_len = len(data)

    sec_max1 = np.empty(ar_len, dtype=data.dtype)
    sec_max2 = np.empty(ar_len, dtype=data.dtype)

    for i in prange(n - 1):
        sec_max1[i] = np.nan

    for sec in prange(ar_len // n):
        s2_max = data[n * sec + n - 1]
        s1_max = data[n * sec + n]

        for i in range(n - 1, -1, -1):
            if data[n * sec + i] > s2_max:
                s2_max = data[n * sec + i]
            sec_max2[n * sec + i] = s2_max

        sec_max1[n * sec + n - 1] = sec_max2[n * sec]

        for i in range(n - 1):
            if n * sec + n + i < ar_len:
                if data[n * sec + n + i] > s1_max:
                    s1_max = data[n * sec + n + i]
                sec_max1[n * sec + n + i] = max(s1_max, sec_max2[n * sec + i + 1])

            else:
                break

    return sec_max1


@njit(parallel=True)
def rolling_min(data, n):
    ar_len = len(data)

    sec_min1 = np.empty(ar_len, dtype=data.dtype)
    sec_min2 = np.empty(ar_len, dtype=data.dtype)

    for i in prange(n - 1):
        sec_min1[i] = np.nan

    for sec in prange(ar_len // n):
        s2_min = data[n * sec + n - 1]
        s1_min = data[n * sec + n]

        for i in range(n - 1, -1, -1):
            if data[n * sec + i] < s2_min:
                s2_min = data[n * sec + i]
            sec_min2[n * sec + i] = s2_min

        sec_min1[n * sec + n - 1] = sec_min2[n * sec]

        for i in range(n - 1):
            if n * sec + n + i < ar_len:
                if data[n * sec + n + i] < s1_min:
                    s1_min = data[n * sec + n + i]
                sec_min1[n * sec + n + i] = min(s1_min, sec_min2[n * sec + i + 1])

            else:
                break

    return sec_min1


@njit
def sma(a, p):
    start = 0
    if np.isnan(a).any():
        start = np.where(np.isnan(a))[0][-1] + 1

    m = np.cumsum(a[start:]) / p
    m[p:] = m[p:] - m[:-p]
    m[:p - 1] = np.nan

    if start > 0:
        m = np.concatenate((a[:start], m))

    return m


@njit(parallel=True)
def smma(arr, p):
    p = p * 2 - 1
    alpha = 2 / (p + 1)

    exp_weights = np.zeros(len(arr))
    exp_weights[p - 1] = np.mean(arr[:p])

    for i in range(p, len(exp_weights) + 1):
        exp_weights[i] = exp_weights[i - 1] * (1 - alpha) + (alpha * (arr[i]))
    exp_weights[:p - 1] = np.nan

    return exp_weights


@njit
def mfi(h, l, c, v, p):
    typical_price = (h + l + c) / 3
    raw_money_flow = typical_price * v

    flow = [typical_price[idx] > typical_price[idx - 1] for idx in prange(1, len(typical_price))]
    flow.insert(0, np.NaN)

    positive_flow = [raw_money_flow[idx] if flow[idx] else 0 for idx in prange(0, len(flow))]
    negative_flow = [raw_money_flow[idx] if not flow[idx] else 0 for idx in prange(0, len(flow))]

    positive_mf = [sum(positive_flow[idx + 1 - p:idx + 1]) for idx in prange(p - 1, len(positive_flow))]
    negative_mf = [sum(negative_flow[idx + 1 - p:idx + 1]) for idx in prange(p - 1, len(negative_flow))]

    positive_mf[:0] = [np.NaN] * (p - 1)
    negative_mf[:0] = [np.NaN] * (p - 1)

    # with np.errstate(divide='ignore'):
    money_ratio = np.array(positive_mf) / np.array(negative_mf)

    return 100 - (100 / (1 + money_ratio))


@jit(forceobj=True)
def trading_allowed(index, start, end):
    hours = np.array([date_time.hour for date_time in index], dtype=np.int32)

    return np.array([True if end >= hour >= start else False for hour in hours], dtype=bool)


@njit(parallel=True)
def vwap(h, l, c, v):
    return np.cumsum(v * (h + l + c) / 3) / np.cumsum(v)


@njit(fastmath=True)
def calc_rsi(array, deltas, avg_gain, avg_loss, p):
    # Use Wilder smoothing method
    up = lambda x: x if x > 0 else 0
    down = lambda x: -x if x < 0 else 0
    i = p + 1
    for d in deltas[p + 1:]:
        avg_gain = ((avg_gain * (p - 1)) + up(d)) / p
        avg_loss = ((avg_loss * (p - 1)) + down(d)) / p
        if avg_loss != 0:
            rs = avg_gain / avg_loss
            array[i] = 100 - (100 / (1 + rs))
        else:
            array[i] = 100
        i += 1

    return array


def rsi(c, p=14):
    deltas = np.append([0], np.diff(c))

    avg_gain = np.sum(deltas[1:p + 1].clip(min=0)) / p
    avg_loss = -np.sum(deltas[1:p + 1].clip(max=0)) / p

    c = np.empty(deltas.shape[0])
    c.fill(np.nan)

    c = calc_rsi(c, deltas, avg_gain, avg_loss, p)

    return c


@njit(parallel=True)
def candlestick_patterns(o, h, l, c):
    direction = np.zeros_like(o)
    concentration_bars = np.zeros_like(o)
    three_line_strike = np.zeros_like(o)

    for i in prange(1, len(o)):
        if o[i-1] < c[i-1]:
            direction[i-1] = 1
        elif o[i-1] > c[i-1]:
            direction[i-1] = -1

    for i in prange(4, len(o)):
        if h[i - 4] > h[i - 3] > h[i - 2] < c[i-1] and l[i - 4] < l[i - 3] < l[i - 2] < l[i-1] and direction[
            i - 3] == -1 and \
                direction[i - 2] == -1:
            concentration_bars[i-1] = 1
        elif h[i - 4] > h[i - 3] > h[i - 2] > h[i-1] and l[i - 4] < l[i - 3] < l[i - 2] > c[i-1] and direction[
            i - 3] == 1 and \
                direction[i - 2] == 1:
            concentration_bars[i-1] = -1

    for i in prange(4, len(o)):
        if direction[i-4] == -1 and direction[i-3] == -1 and direction[i-2] == -1 and direction[i-1] == 1 and o[i-1] \
                <= c[i-2] and c[i-1] > o[i-4] and c[i-4] > c[i-3] > c[i-2]:
            three_line_strike[i-1] = 1
        elif direction[i-4] == 1 and direction[i-3] == 1 and direction[i-2] == 1 and direction[i-1] == -1 and o[i-1] \
                >= c[i-2] and c[i-1] < o[i-4] and c[i-4] < c[i-3] < c[i-2]:
            three_line_strike[i-1] = -1

    return concentration_bars, three_line_strike
