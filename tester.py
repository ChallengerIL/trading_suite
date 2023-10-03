from statistics import mean
from test_trader import Currency
import matplotlib.pyplot as plt
from datetime import datetime
from math import floor
import time as t
import finplot as fplt
from datetime import timezone


class Account:

    def __init__(self, pairs_num, bars_len):
        self.pairs_num = pairs_num
        self.start_balance = 300
        self.balance = self.start_balance * self.pairs_num
        self.balance_max = self.balance
        self.balance_min = self.balance
        self.equity = self.balance
        self.drawdown = 100
        # Include the leverage into the calculation
        self.leverage = 500
        self.lot_size_start = 0.01
        self.lot_size = self.lot_size_start
        self.max_lot_size = 50
        self.currency_converter = 10
        self.profits = []
        self.losses = []
        self.profit_pips = []
        self.loss_pips = []
        self.pips = 0
        self.profit_trades = 0
        self.loss_trades = 0
        self.profit_longs = 0
        self.loss_longs = 0
        self.profit_shorts = 0
        self.loss_shorts = 0
        self.total_trades = 0
        self.trading_period = []
        self.bars_len = bars_len
        self.x = list()
        self.y = list()

    def total_report(self, currency):
        print("Total Report")
        # print(f"Return: {}%")
        if self.profit_trades != 0:
            print("Winning rate: " + str(round(100 / self.total_trades * self.profit_trades)) + "%")
        else:
            print("Winning rate: 0%")
        print("Balance : $" + str(round(self.balance, 2)))
        print("Balance max: $" + str(round(self.balance_max, 2)))
        print("Balance min: $" + str(round(self.balance_min, 2)))
        print("Max drawdown: " + str(round(100 - self.drawdown, 2)) + "%")
        print("Profit trades: " + str(self.profit_trades))
        print("Loss trades: " + str(self.loss_trades))
        print("Number of longs: " + str(self.profit_longs + self.loss_longs))
        print("Profit longs: " + str(self.profit_longs))
        print("Loss longs: " + str(self.loss_longs))
        print("Number of shorts: " + str(self.profit_shorts + self.loss_shorts))
        print("Profit shorts: " + str(self.profit_shorts))
        print("Loss shorts: " + str(self.loss_shorts))
        if len(self.profits) > 0:
            print("Biggest profit: +$" + str(max(self.profits)))
        else:
            print("Biggest profit: $0")
        if len(self.losses) > 0:
            print("Biggest loss: -$" + str(min(self.losses))[1:])
        else:
            print("Biggest loss: $0")
        if len(self.profits) > 0:
            print("Average profit: +$" + str(round(mean(self.profits), 2)))
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
            print(f"Trading rate: {(self.trading_period[1] - self.trading_period[0]) / self.total_trades}")
        else:
            print("Trading rate: 0")
        print("Pips: " + str(self.pips))
        print(f"Testing period: {self.trading_period[1] - self.trading_period[0]}")
        print("\n")

    def balance_plotter(self, dark=True):
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
        ax = fplt.create_plot("Backtesting results", rows=1)

        ax.showGrid(True, True, alpha=0.3)
        labelStyle = {'color': '#FFF', 'font-size': '16pt'}
        ax.setLabel('right', 'Balance', **labelStyle)
        ax.setLabel('bottom', 'Date', **labelStyle)

        fplt.plot(self.x, self.y, width=2, color="white", ax=ax)
        fplt.show()


def backtester(strategy: dict, params: list, plotting=False, verbose=False, cushion=2, **kw):
    pairs = strategy['pairs']
    if 'pairs' in kw:
        pairs = kw['pairs']

    start_pos = list()
    lengths = list()
    pairs_to_test = list()

    for i in range(len(pairs)):
        start = t.time()
        # pairs_to_test.append(Currency(pair=pairs[i], strategy=strategy,
        # params=params[i], verbose=verbose, cushion=cushion))
        pairs_to_test.append(
            Currency(pair=pairs[i], strategy=strategy, params=params, verbose=verbose, cushion=cushion)
        )
        lengths.append(len(pairs_to_test[i].open))
        start_pos.append(pairs_to_test[i].start_pos)
        end = t.time()
        print("Elapsed (with compilation) = %s" % (end - start))
        print("\n")

    account = Account(pairs_num=len(pairs), bars_len=min(lengths))
    account.trading_period.append(pairs_to_test[0].index[max(start_pos) + max(lengths) - min(lengths) + 2])

    for i in range(max(start_pos) + max(lengths) - min(lengths) + 2, min(lengths)):
        if account.balance > 3 * account.lot_size * 100:

            # Dynamically adjust the lot size based on the current balance
            if pairs_to_test[0].adjust_lot_size:
                # If the current balance is greater than the initial balance times 2
                # (multiplied by the number of trading pairs)
                if account.balance >= (account.start_balance * 2) * len(pairs):
                    # If the latest lot size value is less than the maximum lot size allowed
                    if account.lot_size < account.max_lot_size:
                        if round(floor(account.balance / len(pairs)) / (account.start_balance * 100),
                                 2) > account.lot_size:
                            account.lot_size = round(floor(account.balance / len(pairs)) / (account.start_balance * 100)
                                                     , 2)

            for pair in pairs_to_test:
                pair.tester(
                    account=account,
                    index=i,
                )

            if account.balance > account.balance_max:
                account.balance_max = account.balance

            if account.balance < account.balance_min:
                account.balance_min = account.balance

            if 100 / account.balance_max * account.balance < account.drawdown:
                account.drawdown = 100 / account.balance_max * account.balance

            if plotting:
                account.x.append(pairs_to_test[0].index[i])
                account.y.append(account.balance)
        else:
            if plotting:
                account.x.append(pairs_to_test[0].index[i])
                account.y.append(account.balance)

            account.trading_period.append(pairs_to_test[0].index[i])
            print("Not enough free margin")
            print("\n")
            break

    if len(account.trading_period) < 2:
        account.trading_period.append(pairs_to_test[0].index[min(lengths) - 1])

    if plotting:
        if account.total_trades > 0:
            [pair.total_report(account) for pair in pairs_to_test]
            account.total_report(pairs_to_test[0])
            account.balance_plotter()
        else:
            print("No trades.")

        [pair.plotter() for pair in pairs_to_test]

    if len(account.profits) > 0:
        profits = round(mean(account.profits), 2)
    else:
        profits = 0

    if len(account.losses) > 0:
        losses = round(mean(account.losses), 2)
    else:
        losses = 0

    if account.total_trades > 0 and account.profit_trades > 0:
        win_rate = round(100 / account.total_trades * account.profit_trades)
    else:
        win_rate = 0

    results = win_rate, round(100 - account.drawdown, 2), profits, losses, account.total_trades, account.pips, \
        round(account.balance, 2)

    return results
