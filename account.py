import metatrader
import pandas as pd
# from statistics import mean
from datetime import date, datetime
import os
from termcolor import colored
import sys
from config import SYMBOL_HEAD, SYMBOL_TAIL, STRATEGIES, FILES_DIR, SLEEP_DAY, \
    WEEKEND_SLEEP
# import time as t

ENTRY_L = 2
ENTRY_S = 3

if os.name == 'nt':
    import ctypes

    class _CursorInfo(ctypes.Structure):
        _fields_ = [("size", ctypes.c_int),
                    ("visible", ctypes.c_byte)]


def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def hide_cursor():
    if os.name == 'nt':
        ci = _CursorInfo()
        handle = ctypes.windll.kernel32.GetStdHandle(-11)
        ctypes.windll.kernel32.GetConsoleCursorInfo(handle, ctypes.byref(ci))
        ci.visible = False
        ctypes.windll.kernel32.SetConsoleCursorInfo(handle, ctypes.byref(ci))
    elif os.name == 'posix':
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()


class Account(object):

    def __init__(self, data, data_time, strategy, logger=None):
        self.strategy = strategy
        self.mt5 = metatrader.mt5
        self.account_info = self.mt5.account_info()
        self.orders = self.mt5.positions_get()
        self.active_orders = {'time': list(), 'symbol': list(), 'ticket': list(), 'type': list(), 'volume': list(),
                              'entry_price': list(), 'sl': list(), 'tp': list(), 'price': list(), 'swap': list(),
                              'state': list(), 'profit': list(), 'loss': list()}
        self.update_all()
        self.balance = self.account_info.balance
        self.balance_max = data["Account"]["balance_max"]
        self.balance_min = data["Account"]["balance_min"]
        self.equity = self.account_info.equity
        self.margin_free = self.account_info.margin_free
        self.drawdown = round(data["Account"]["drawdown"], 2)
        self.pips = data["Account"]["pips"]
        self.leverage = self.account_info.leverage
        self.lot_size = data["Account"]["lot_size"]
        self.bars_passed = data["Account"]["bars_passed"]
        self.account_profit = self.account_info.profit
        self.profits = data["Account"]["profits"]
        self.losses = data["Account"]["losses"]
        self.profit_trades = data["Account"]["profit_trades"]
        self.loss_trades = data["Account"]["loss_trades"]
        self.profit_longs = data["Account"]["profit_longs"]
        self.loss_longs = data["Account"]["loss_longs"]
        self.profit_shorts = data["Account"]["profit_shorts"]
        self.loss_shorts = data["Account"]["loss_shorts"]
        self.total_trades = data["Account"]["total_trades"]
        self.active_positions = self.mt5.positions_total()
        self.last_params_update = date.fromisoformat(data_time)
        self.rsi_vwap_values = list()
        self.last_index = None
        self.last_bar_time = None
        self.pairs = self.strategy["pairs"]
        self.logger = logger

    def prepare_request(self, direction, action_type, symbol, price, **kw):
        open_orders = {"long": self.mt5.ORDER_TYPE_BUY, "short": self.mt5.ORDER_TYPE_SELL}
        close_orders = {"long": self.mt5.ORDER_TYPE_SELL, "short": self.mt5.ORDER_TYPE_BUY}

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "type": open_orders[direction],
            "price": price,
            "deviation": kw["deviation"],
            "magic": 1,
            "comment": kw["comment"],
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_RETURN,
        }

        if action_type == "close":
            request["type"] = close_orders[direction]

        if "tp" in kw:
            if direction == "long":
                request["tp"] = price + kw["tp"] * kw["point"]
            else:
                request["tp"] = price - kw["tp"] * kw["point"]

        if "sl" in kw:
            if direction == "long":
                request["sl"] = price - kw["sl"] * kw["point"]
            else:
                request["sl"] = price + kw["sl"] * kw["point"]

        if "position_id" in kw:
            request["position"] = kw["position_id"]

        if "lot_size" not in kw:
            request["volume"] = self.lot_size

        return request

    def place_order(self, direction, symbol, price, deviation, currency_logger, **kw):
        result = self.mt5.order_send(self.prepare_request(
            direction=direction, action_type="open", symbol=symbol, price=price, deviation=deviation,
            comment=f"Going {direction} on {symbol}", **kw)
        )

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            currency_logger.error(f"Failed to go {direction} on {symbol} for {price}, retcode={result.retcode}")

            if result.retcode == 10004 or result.retcode == 10021:
                currency_logger.info("Trying again....")
                if direction == "long":
                    price = self.mt5.symbol_info_tick(symbol).ask
                else:
                    price = self.mt5.symbol_info_tick(symbol).bid
                self.place_order(direction, symbol, price, currency_logger, **kw)
        else:
            currency_logger.info(f"Successfully went {direction} on {symbol} for {price}")

    def close_order(self, direction, strategy, symbol, position_id, price, lot_size, deviation, currency_logger):
        result = self.mt5.order_send(self.prepare_request(
            direction=direction, action_type="close", symbol=symbol, price=price, deviation=deviation,
            comment=f"Closing {direction} on {symbol}", position_id=position_id, lot_size=lot_size))

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            currency_logger.info(
                f"Failed to close {direction} {position_id} {symbol} at {price}, retcode={result.retcode}"
            )
            if result.retcode == 10004 or result.retcode == 10021:
                currency_logger.info("Trying again....")
                if direction == "long":
                    price = self.mt5.symbol_info_tick(symbol).bid
                else:
                    price = self.mt5.symbol_info_tick(symbol).ask
                self.close_order(
                    direction, strategy, symbol, position_id, price, lot_size, deviation, currency_logger
                )
        else:
            currency_logger.info(f"Successfully closed {direction} on {symbol} for {price}")

    def update_call(self):
        # Try switching to timedelta (current index from data >= last_update_index + timedelta(minutes=1))

        self.last_bar_time = self.mt5.copy_rates_from_pos(SYMBOL_HEAD + self.strategy["pairs"][0] + SYMBOL_TAIL,
                                                          TFS_DICT[self.strategy["tf"]], 0, 1)
        self.last_bar_time = pd.DataFrame(self.last_bar_time)
        self.last_bar_time['time'] = pd.to_datetime(self.last_bar_time['time'], unit='s')
        self.last_bar_time = self.last_bar_time['time'][0]

        if self.last_bar_time != self.last_index:
            return True

    def data_update(self):
        self.last_index = self.last_bar_time

        self.account_info = self.mt5.account_info()
        self.balance = self.account_info.balance
        self.equity = self.account_info.equity
        self.margin_free = self.account_info.margin_free
        self.account_profit = self.account_info.profit
        self.active_positions = self.mt5.positions_total()

        if self.balance > self.balance_max:
            self.balance_max = self.balance

        if self.balance < self.balance_min:
            self.balance_min = self.balance

        if round(100 - 100 / self.balance_max * self.equity, 2) > self.drawdown:
            self.drawdown = round(100 - 100 / self.balance_max * self.equity, 2)

        updated_account_data = {
            'Account': {"account profit": self.account_profit, "balance": self.balance, "equity": self.equity,
                        "free margin": self.margin_free, "lot_size": self.lot_size, "balance_max": self.balance_max,
                        "balance_min": self.balance_min, "drawdown": self.drawdown, "pips": self.pips,
                        "bars_passed": self.bars_passed, "profits": self.profits, "losses": self.losses,
                        "profit_trades": self.profit_trades, "loss_trades": self.loss_trades,
                        "profit_longs": self.profit_longs, "loss_longs": self.loss_longs,
                        "profit_shorts": self.profit_shorts, "loss_shorts": self.loss_shorts,
                        "total_trades": self.total_trades, "active positions": self.active_positions,
                        "last_params_update": self.last_params_update, "last_data_update": self.last_index
                        }
        }

        for i in range(len(self.rsi_vwap_values)):
            updated_account_data['Account'][self.strategy["pairs"][i]] = self.rsi_vwap_values[i]

        updated_account_df = pd.DataFrame(data=updated_account_data)
        updated_account_df.to_csv(FILES_DIR + "account_data.csv")

    def update_schedule(self):
        if date.today().weekday() == SLEEP_DAY and WEEKEND_SLEEP:
            os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
        if date.today().weekday() == self.strategy["update_weekday"]:
            if (date.today() - self.last_params_update).days > self.strategy["update_period"] and \
                    datetime.now().time() > self.strategy["update_time"]:
                self.logger.info("Started updating trading parameters")
                for pair in self.pairs:
                    os.system(f"python ga.py {pair} main.py")

                self.last_params_update = date.today()
                self.data_update()
                self.logger.info("Finished updating trading parameters")

                params_data = pd.read_csv(FILES_DIR + "params_data.csv", index_col=0)
                for n in self.pairs:
                    params_data[n.upper()] = params_data[n.upper()].apply(eval)

    def update_all(self):
        self.orders = self.mt5.positions_get()
        self.active_orders = {'time': list(), 'symbol': list(), 'ticket': list(), 'type': list(), 'volume': list(),
                              'entry_price': list(), 'sl': list(), 'tp': list(), 'price': list(), 'swap': list(),
                              'state': list(), 'profit': list(), 'loss': list()}

        for i in range(len(self.orders)):
            self.active_orders['time'].append(datetime.utcfromtimestamp(int(self.orders[i].time_update)).strftime(
                '%Y-%m-%d %H:%M:%S'))
            self.active_orders['symbol'].append(str(self.orders[i].symbol))
            self.active_orders['ticket'].append(self.orders[i].ticket)
            self.active_orders['type'].append(self.orders[i].type)
            self.active_orders['volume'].append(str(self.orders[i].volume))
            self.active_orders['entry_price'].append(str(round(self.orders[i].price_open, 5)))
            self.active_orders['sl'].append(str(round(self.orders[i].sl, 5)))
            self.active_orders['tp'].append(str(round(self.orders[i].tp, 5)))
            self.active_orders['price'].append(round(self.orders[i].price_current))
            self.active_orders['swap'].append(self.orders[i].swap)
            self.active_orders['state'].append(round(self.orders[i].profit, 2))
            if self.orders[i].type == 0:
                if self.orders[i].tp != 0:
                    self.active_orders['profit'].append(round(self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_BUY,
                                                                                         self.orders[i].symbol,
                                                                                         self.orders[i].volume,
                                                                                         self.orders[i].price_open,
                                                                                         self.orders[i].tp), 2))
                else:
                    self.active_orders['profit'].append(0)
                if self.orders[i].sl != 0:
                    self.active_orders['loss'].append(round(self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_BUY,
                                                                                       self.orders[i].symbol,
                                                                                       self.orders[i].volume,
                                                                                       self.orders[i].price_open,
                                                                                       self.orders[i].sl), 2))
                else:
                    self.active_orders['loss'].append(0)
            elif self.orders[i].type == 1:
                if self.orders[i].tp != 0:
                    self.active_orders['profit'].append(round(self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_SELL,
                                                                                         self.orders[i].symbol,
                                                                                         self.orders[i].volume,
                                                                                         self.orders[i].price_open,
                                                                                         self.orders[i].tp), 2))
                else:
                    self.active_orders['profit'].append(0)
                if self.orders[i].sl != 0:
                    self.active_orders['loss'].append(round(self.mt5.order_calc_profit(self.mt5.ORDER_TYPE_SELL,
                                                                                       self.orders[i].symbol,
                                                                                       self.orders[i].volume,
                                                                                       self.orders[i].price_open,
                                                                                       self.orders[i].sl), 2))
                else:
                    self.active_orders['loss'].append(0)

        for i in range(len(self.active_orders['type'])):
            if self.active_orders['type'][i] == 0:
                self.active_orders['type'][i] = 'buy'
            elif self.active_orders['type'][i] == 1:
                self.active_orders['type'][i] = 'sell'

    def display(self, params):
        margin = {'time': 1, 'symbol': 1, 'ticket': 1, 'type': 6, 'volume': 1, 'entry_price': 8, 'sl': 8, 'tp': 1,
                  'price': 8, 'swap': 5, 'state': 6, 'profit': 6, 'loss': 6}

        account_info = self.mt5.account_info()

        update_call = self.mt5.copy_rates_from_pos(SYMBOL_HEAD + self.strategy["pairs"][0] + SYMBOL_TAIL,
                                                   TFS_DICT["M1"], 0, 1)
        update_call = pd.DataFrame(update_call)
        update_call['time'] = pd.to_datetime(update_call['time'], unit='s')
        update_call = update_call['time'][0]

        top_row = "Last update: {} | Balance: ${} | Equity: ${} | Free Margin: ${}".format(
            update_call, str(account_info.balance), str(account_info.equity), str(account_info.margin_free)
        )
        pairs_row = " " * 5
        value_row = "Value"
        short_row = "Short"
        long_row = "Long "
        headers_row = " "*8 + "Time" + " "*10 + "Pair" + " "*4 + "Ticket" + " "*3 + "Type" + " "*1 + "Volume" + " "*1\
                      + "Price" + " "*5 + "SL" + " "*6 + "TP" + " "*4 + "Price" + " "*3 + "Swap" + " "*1 + "State" +\
                      " "*1 + "Profit" + " "*3 + "Loss"
        updated_orders = self.mt5.positions_get()
        total_orders = self.mt5.positions_total()
        total_state = round(sum(self.active_orders['state']) + sum(self.active_orders['swap']), 2)
        total_profit = round(sum(self.active_orders['profit']) + sum(self.active_orders['swap']), 2)
        total_loss = round(sum(self.active_orders['loss']) + sum(self.active_orders['swap']), 2)

        for i in range(0, len(self.strategy["pairs"])):
            pairs_row += " " * 1 + self.strategy["pairs"][i]
            value_row += " " * (7-len(str(self.rsi_vwap_values[i-1]))) + str(self.rsi_vwap_values[i])
            short_row += " " * (7-len(str(params[self.strategy["pairs"][i-1]]['params'][ENTRY_S]))) + \
                         str(params[self.strategy["pairs"][i]]['params'][ENTRY_S])
            long_row += " " * (7-len(str(params[self.strategy["pairs"][i-1]]['params'][ENTRY_L]))) + \
                        str(params[self.strategy["pairs"][i]]['params'][ENTRY_L])

        print(top_row + "\n" + pairs_row + "\n" + value_row + "\n" + short_row + "\n" + long_row + "\n"*2 + headers_row)

        if len(self.active_orders['ticket']) > 0:
            if len(updated_orders) == len(self.orders) and self.active_orders['ticket'][-1] == \
                    updated_orders[-1].ticket:
                for i in range(len(self.orders)):
                    self.active_orders['state'][i] = round(updated_orders[i].profit, 2)
                    self.active_orders['price'][i] = round(updated_orders[i].price_current, 5)
            else:
                self.update_all()
        else:
            self.update_all()

        for i in reversed(range(len(self.active_orders['symbol']))):
            print(self.active_orders['time'][i] + " "*margin['time'], end="", flush=True)
            if self.active_orders['loss'][i] + self.active_orders['swap'][i] - self.active_orders['state'][i] > -1:
                print(colored(self.active_orders['symbol'][i], 'red') + " "*margin['symbol'], end="", flush=True)
            elif self.active_orders['profit'][i] + self.active_orders['swap'][i] - self.active_orders['state'][i] < 1:
                print(colored(self.active_orders['symbol'][i], 'green') + " " * margin['symbol'], end="", flush=True)
            else:
                print(self.active_orders['symbol'][i] + " " * margin['symbol'], end="", flush=True)
            print(str(self.active_orders['ticket'][i]) + " "*margin['ticket'], end="", flush=True)
            print(self.active_orders['type'][i] + " "*(margin['type']-len(self.active_orders['type'][i])), end="",
                  flush=True)
            print(self.active_orders['volume'][i] + " "*margin['volume'], end="", flush=True)
            print(self.active_orders['entry_price'][i] + " " *
                  (margin['entry_price']-len(self.active_orders['entry_price'][i])), end="", flush=True)
            print(self.active_orders['sl'][i] + " " * (margin['sl']-len(self.active_orders['tp'][i])),
                  end="", flush=True)
            print(self.active_orders['tp'][i] + " "*(margin['price']-len(str(self.active_orders['tp'][i]))),
                  end="", flush=True)
            print(str(self.active_orders['price'][i]) + " " *
                  (margin['price']-len(str(self.active_orders['price'][i]))), end="", flush=True)
            print(" " * (margin['swap']-len(str(self.active_orders['swap'][i]))) + str(self.active_orders['swap'][i])
                  + " " * 1, end="", flush=True)
            if self.active_orders['state'][i] > 0:
                print(" " + colored(str(round(self.active_orders['state'][i], 2)), 'green'), end="", flush=True)
            elif self.active_orders['state'][i] + self.active_orders['swap'][i] < 0:
                print(colored(str(round(self.active_orders['state'][i] + self.active_orders['swap'][i], 2)),
                              'red') + " "*1, end="", flush=True)
            else:
                print(" "+str(round(self.active_orders['state'][i] + self.active_orders['swap'][i], 2)), end="",
                      flush=True)
            print(" "*(margin['profit']-len(str(self.active_orders['state'][i]))) +
                  str(round(self.active_orders['profit'][i] + self.active_orders['swap'][i], 2)) + " "*1, end="",
                  flush=True)
            print(" "*(margin['loss']-len(str(self.active_orders['profit'][i]))) +
                  str(round(self.active_orders['loss'][i] + self.active_orders['swap'][i], 2)))

        print("Total orders: {}".format(total_orders) + " "*(79-len(str(total_orders))-len(str(total_state))),
              end="", flush=True)
        if total_state > 0:
            print(colored(str(total_state), 'green') + " " * 1, end="", flush=True)
        elif total_state < 0:
            print(colored(str(total_state), 'red') + " " * 1, end="", flush=True)
        else:
            print(str(total_state) + " " * 1, end="", flush=True)
        print(" "*(10-len(str(total_state))-len(str(total_profit))) + str(total_profit) + " " * 1, end="", flush=True)
        print(" "*(margin['loss']-len(str(total_profit))) + str(total_loss))
