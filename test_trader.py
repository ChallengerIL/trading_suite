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
from strategies import strategy_runner


@njit
def pip_calc(start_price, end_price, multiplier, spread=0):
    return (end_price - start_price) / multiplier - spread


class Currency:

    def __init__(self, pair, strategy, verbose=False, cushion=2, **kw):
        self.name = pair
        self.verbose = verbose
        self.params = kw["params"]
        self.strategy = strategy
        # self.start_pos = round(max(self.params[2:8]))
        self.start_pos = 0
        all_tfs = list()
        for k, v in self.strategy['indicators'].items():
            all_tfs += list(v.keys())

        self.tfs = sorted(list(set(all_tfs)), key=self.strategy['tfs'].index)

        with h5py.File(f"{FILES_DIR}pairs_data.h5", 'r') as f:
            self.df = {tf: np.array(f[f"{tf}_{pair}"]) for tf in self.tfs}

        self.config = self.account_data = pd.read_csv(FILES_DIR + "account_data.csv", index_col=0)

        # Main
        self.index = index_to_datetime(self.df[self.tfs[0]]["time"])
        self.open = self.df[self.tfs[0]]['open']
        self.high = self.df[self.tfs[0]]['high']
        self.low = self.df[self.tfs[0]]['low']
        self.close = self.df[self.tfs[0]]['close']
        self.vol = self.df[self.tfs[0]]['vol']
        self.spread = self.df[self.tfs[0]]['spread'] / 10

        if str(self.open[-1]).index(".") >= 3:  # JPY pair
            self.multiplier = 0.01
        else:
            self.multiplier = 0.0001

        self.tp = 20
        self.sl = 200
        self.tp_multiplier = 1.5
        self.slope_level = 0.0002
        self.break_even = 25
        self.atr_multiplier = 5  # 1-30
        self.atr_period = 21  # 10-100
        self.trailing_stop = 100

        self.indicators = list()

        for indicator in self.strategy['indicators']:
            setattr(self, indicator, Indicator(
                name=indicator, df=self.df, strategy=self.strategy, tfs=self.tfs
            ))

            indi = getattr(self, indicator)

            self.indicators.append(indi)

        self.lot_size = 0.01
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
        self.win_series = 0
        self.fast_growth = [3, 4, 5, 7, 9, 11, 14, 19, 24, 32, 41, 54, 70, 91, 118, 154, 200, 260, 337, 439, 570,
                            741, 964, 1253, 1628, 2117, 2752, 3578, 4651, 6046]
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
        self.direction_won = 0
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

    def tester(self, account, index):
        strategy_runner(self, account, index)

    def open_long(self, index, custom_tp=None, custom_sl=None, sl_multiplied_tp=None,
                  previous_tp=False, previous_sl=False):
        open_price = self.open[index]

        price = open_price + self.multiplier * (self.spread[index - 1] + self.cushion)

        tp = price + self.tp * self.multiplier
        sl = price - self.sl * self.multiplier

        # if custom_tp is not None:
        #     tp = custom_tp
        #
        # if custom_sl is not None:
        #     sl = custom_sl
        #
        # if sl_multiplied_tp is not None:
        #     tp = price + (price - sl) * sl_multiplied_tp

        if previous_tp:
            if len(self.long_tps) > 0:
                self.tp = self.long_tps[-1]

        if previous_sl:
            if len(self.long_sls) > 0:
                self.sl = self.long_sls[-1]

        self.long_tps.append(tp)
        self.long_sls.append(sl)
        self.active_longs.append(price)
        self.long_lot_sizes.append(self.lot_size)
        self.went_long_price[index] = price

        if self.verbose:
            print(f"Going Long on {self.name}, {self.lot_size}")
            print(self.index[index])
            print(f"Take Profit: {tp}")
            print(f"Stop Loss: {sl}")
            print("Active Longs: " + str(self.active_longs))
            print("\n")

    def open_short(self, index, custom_tp=None, custom_sl=None, sl_multiplied_tp=None,
                   previous_tp=False, previous_sl=False):
        open_price = self.open[index]

        tp = open_price - self.tp * self.multiplier
        sl = open_price + self.sl * self.multiplier

        # if custom_tp is not None:
        #     tp = custom_tp
        #
        # if custom_sl is not None:
        #     sl = custom_sl
        #
        # if sl_multiplied_tp is not None:
        #     tp = open_price - (sl - open_price) * sl_multiplied_tp

        if previous_tp:
            if len(self.short_tps) > 0:
                tp = self.short_tps[-1]

        if previous_sl:
            if len(self.short_sls) > 0:
                sl = self.short_sls[-1]

        self.active_shorts.append(open_price)
        self.short_tps.append(tp)
        self.short_sls.append(sl)
        self.short_lot_sizes.append(self.lot_size)
        self.went_short_price[index] = open_price

        if self.verbose:
            print(f"Going Short on {self.name}, {self.lot_size}")
            print(self.index[index])
            print(f"Take Profit: {tp}")
            print(f"Stop Loss: {sl}")
            print("Active Shorts: " + str(self.active_shorts))
            print("\n")

    def check_exit(self, account, index, close_long=False, close_short=False, martin=False, break_even=False,
                   trailing_stop=False, atr_trailing=False, fast_growth=False, fixed_tp_sl=False):

        # Add Break Even mode with a fixed tp of ~ 1 pip after price moved away 20+ pips

        if not martin or not fast_growth:
            if self.lot_size < account.lot_size:
                self.lot_size = account.lot_size

        if fast_growth:
            self.adjust_lot_size = False
            self.lot_size = self.fast_growth[self.win_series] / 100

        if len(self.active_longs) > 0:
            for i, value in enumerate(self.active_longs):

                account.open_positions_state.append(
                    round(round(pip_calc(value, self.low[index-1], self.multiplier), 2)
                          * (self.long_lot_sizes[i] * account.currency_converter), 2)
                )

                current_price = self.open[index]
                pips = 0

                if atr_trailing:
                    if current_price - self.atr[index - 1] * self.atr_multiplier > self.long_sls[i]:
                        self.long_sls[i] = current_price - self.atr[index - 1] * self.atr_multiplier

                if trailing_stop:
                    if current_price - self.trailing_stop * self.multiplier > self.long_sls[i]:
                        self.long_sls[i] = current_price - self.trailing_stop * self.multiplier

                if fixed_tp_sl:
                    if self.high[index - 1] >= self.long_tps[i] or current_price >= self.long_tps[i]:
                        pips = round(pip_calc(value, self.long_tps[i], self.multiplier), 2)

                    elif self.low[index - 1] <= self.long_sls[i] or current_price <= self.long_sls[i]:
                        pips = round(pip_calc(value, self.long_sls[i], self.multiplier), 2)

                if current_price >= self.long_tps[i] or current_price <= self.long_sls[i] or close_long or pips != 0:
                    if not fixed_tp_sl:
                        pips = round(pip_calc(value, current_price, self.multiplier), 2)

                    result = round(pips * (self.long_lot_sizes[i] * account.currency_converter), 2)

                    account.open_positions_state.pop()

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
                        if fixed_tp_sl:
                            self.closed_long_profit[index-1] = current_price
                        else:
                            self.closed_long_profit[index] = current_price
                        if martin:
                            self.lot_size = account.lot_size
                        self.direction_lost = 0
                        self.direction_won = 1
                        if fast_growth:
                            if self.win_series < len(self.fast_growth) - 1:
                                self.win_series += 1
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
                        if fixed_tp_sl:
                            self.closed_long_loss[index-1] = current_price
                        else:
                            self.closed_long_loss[index] = current_price
                        if martin:
                            self.lot_size *= 2
                        self.direction_lost = 1
                        self.direction_won = 0
                        if fast_growth:
                            if self.win_series > 0:
                                self.win_series -= 1

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

                account.open_positions_state.append(
                    round(round((value - self.high[index-1] + (self.spread[index - 1] + self.cushion) *
                                 self.multiplier) / self.multiplier, 2) *
                          (self.short_lot_sizes[i] * account.currency_converter), 2)
                )

                current_price = self.open[index] + (self.spread[index - 1] + self.cushion) * self.multiplier
                pips = 0

                if atr_trailing:
                    if current_price + self.atr[index - 1] * self.atr_multiplier < self.short_sls[i]:
                        self.short_sls[i] = current_price + self.atr[index - 1] * self.atr_multiplier

                if trailing_stop:
                    if current_price + self.trailing_stop * self.multiplier < self.short_sls[i]:
                        self.short_sls[i] = current_price + self.trailing_stop * self.multiplier

                if fixed_tp_sl:
                    if self.low[index - 1] + (self.spread[index - 1] + self.cushion) * self.multiplier <= \
                            self.short_tps[i] \
                            or current_price <= self.short_tps[i]:
                        pips = round((value - self.short_tps[i]) / self.multiplier, 2)
                    elif self.high[index - 1] + (self.spread[index - 1] + self.cushion) * self.multiplier >= \
                            self.short_sls[i] or current_price >= self.short_sls[i]:
                        pips = round((value - self.short_sls[i]) / self.multiplier, 2)

                if current_price <= self.short_tps[i] or current_price >= self.short_sls[i] or close_short or pips != 0:
                    if not fixed_tp_sl:
                        pips = round((value - current_price) / self.multiplier, 2)

                    result = round(pips * (self.short_lot_sizes[i] * account.currency_converter), 2)

                    account.open_positions_state.pop()

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
                        if fixed_tp_sl:
                            self.closed_short_profit[index-1] = current_price
                        else:
                            self.closed_short_profit[index] = current_price
                        if martin:
                            self.lot_size = account.lot_size
                        self.direction_lost = 0
                        self.direction_won = -1
                        if fast_growth:
                            if self.win_series < len(self.fast_growth) - 1:
                                self.win_series += 1
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
                        if fixed_tp_sl:
                            self.closed_short_loss[index-1] = current_price
                        else:
                            self.closed_short_loss[index] = current_price
                        if martin:
                            self.lot_size *= 2
                        self.direction_lost = -1
                        self.direction_won = 0
                        if fast_growth:
                            if self.win_series > 0:
                                self.win_series -= 1

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
        ax = fplt.create_plot(self.name, rows=self.strategy["plotting_rows"], init_zoom_periods=200)

        def plot_atr(ax):
            fplt.plot(x, self.atr_upper, legend="Upper ATR", style="...", color="blue", ax=ax)
            fplt.plot(x, self.atr_lower, legend="Lower ATR", style="...", color="blue", ax=ax)

        def plot_overlay_bars():
            overlay_plot = fplt.candlestick_ochl(
                [x, self.overlay_open, self.overlay_close, self.overlay_high, self.overlay_low]
            )
            overlay_plot.colors.update(dict(bull_body='#bfb', bull_shadow='#ada', bear_body='#fbc', bear_shadow='#dab'))

        def plot_slope(ax, data_arr, title, period, color):
            fplt.add_line((0, 0), (len(x), 0), color='#fff', style='---', ax=ax)
            fplt.plot(x, slope(data_arr, period), legend=title, color=color, ax=ax)

        def plot_wpr_trend():
            fplt.plot(x, self.wpr_up_lines, legend="WPR Trend Up", width=marker_size, color="blue", ax=ax[0])
            fplt.plot(x, self.wpr_down_lines, legend="WPR Trend Down", width=marker_size, color="red", ax=ax[0])

        def plot_markers():
            fplt.plot(x, self.went_short_price, style='v', width=marker_size, legend="Went Short", color='purple',
                      ax=ax[0])
            fplt.plot(x, self.went_long_price, style='^', width=marker_size, legend="Went Long", color='purple',
                      ax=ax[0])
            fplt.plot(x, self.closed_long_profit, style='^', width=marker_size, legend="Closed Long Profit",
                      color='green', ax=ax[0])
            fplt.plot(x, self.closed_long_loss, style='^', width=marker_size, legend="Closed Long Loss",
                      color='red', ax=ax[0])
            fplt.plot(x, self.closed_short_profit, style='v', width=marker_size, legend="Closed Short Profit",
                      color='green', ax=ax[0])
            fplt.plot(x, self.closed_short_loss, style='v', width=marker_size, legend="Closed Short Loss",
                      color='red', ax=ax[0])

        [indicator.plot(x, ax) for indicator in self.indicators]

        # plot_atr(ax1)
        # plot_slope(ax4, self.)
        # plot_wpr_trend()
        # plot_overlay_bars()
        plot_markers()

        # fplt.plot(x, self.hma.m1_80_slope - self.hma.m1_12_slope, color='white', ax=ax[1])
        # fplt.set_y_range(-0.001, 0.001, ax=ax[1])

        # for n in range(self.high_peaks.shape[1]):
        #     fplt.plot(x, self.high_peaks[:, n], color='green', ax=ax1)
        #     fplt.plot(x, self.low_peaks[:, n], color='blue', ax=ax1)

        # fplt.plot(x, self.peaks_up, style="o", color='green', ax=ax1)
        # fplt.plot(x, self.peaks_down, style="o", color='red', ax=ax1)

        # fplt.plot(x, self.direction, color="white", ax=ax2)

        # fplt.add_line((0, 20), (len(x), 20), color='#fff', style='---', ax=ax3)
        # fplt.add_line((0, 80), (len(x), 80), color='#fff', style='---', ax=ax3)
        # fplt.set_y_range(0, 100, ax=ax3)
        # fplt.set_y_range(-1, 1, ax=ax2)

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
        fplt.candlestick_ochl([x, self.open, self.close, self.high, self.low], ax=ax[0])
        # fplt.candlestick_ochl([x, self.open_ha, self.close_ha, self.high_ha, self.low_ha], ax=ax1)

        fplt.show()
