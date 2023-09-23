import time as t
import logging
from datetime import date
import account as acc
from trader import Currency
from metatrader import connect, shutdown
import pandas as pd
from config import FILES_DIR, SLEEP_DAY, WEEKEND_SLEEP


class Bot:

    def __init__(self, strategy):
        self.strategy = strategy
        logging.basicConfig(filename=FILES_DIR + "the_bot.log", format='%(asctime)s %(message)s', filemode='a')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        self.account_data = pd.read_csv(FILES_DIR + "account_data.csv", index_col=0)
        self.params_data = pd.read_csv(FILES_DIR + "params_data.csv", index_col=0)
        self.params_data = self.params_data.drop(['update_date'], axis=0)
        # self.trading_data = pd.read_csv(FILES_FOLDER + "trading_data.csv", index_col=0)
        for n in self.strategy["pairs"]:
            # self.trading_data[n] = self.trading_data[n].apply(eval)
            print(n)
            print(self.params_data[n])
            self.params_data[n] = self.params_data[n].apply(eval)
        self.account_data_time = self.account_data['Account']['last_params_update']
        self.account_data = self.account_data.drop(['last_params_update', 'last_data_update'], axis=0)
        self.account_data['Account'] = self.account_data['Account'].apply(eval)
        self.active = True
        connect(self.logger)
        self.account = acc.Account(self.account_data, self.account_data_time, self.strategy,
                                   self.logger)
        acc.hide_cursor()
        self.trading_pairs = list()

    def run(self):
        while self.active:
            try:
                if self.account.update_call():
                    [self.trading_pairs.append(Currency(
                        pair=pair, strategy=self.strategy, params=self.params_data[pair]['params'], logger=self.logger))
                     for pair in self.strategy["pairs"]]

                    if self.ready_to_trade():
                        for i in range(-1, 0):
                            if self.account.equity > 3 and self.account.last_index != self.account.last_bar_time:

                                self.account.bars_passed += 1

                                if self.account.equity >= int(len(self.strategy["pairs"]) * 200):
                                    if (self.account.equity // (len(self.strategy["pairs"]) * 200)) / 100 > \
                                            self.account.lot_size:
                                        self.account.lot_size = (self.account.equity // (len(self.strategy["pairs"])
                                                                                         * 200)) / 100

                                for pair in self.trading_pairs:
                                    pair.m15_rsi_vwap(
                                        account=self.account,
                                        index=i,
                                        lot_size=self.account.lot_size,
                                        equity=self.account.equity,
                                    )

                                self.account.rsi_vwap_values = list()

                                for pair in self.trading_pairs:
                                    self.account.rsi_vwap_values.append(round(pair.df.rsi_vwap[i-1], 2))

                                self.account.data_update()
                                self.account.last_index = self.account.last_bar_time
                                self.trading_pairs = list()
                    else:
                        self.trading_pairs = list()

                self.account.update_schedule()
                acc.clear()
                self.account.display(self.params_data)
                t.sleep(1)
                if date.today().weekday() == SLEEP_DAY and WEEKEND_SLEEP:
                    self.active = False

            except KeyboardInterrupt:
                self.active = False

        self.account.data_update()
        shutdown(self.logger)

    def ready_to_trade(self):
        for i in range(0, len(self.trading_pairs)):
            if self.trading_pairs[i - 1].index[-1] != self.trading_pairs[i].index[-1]:
                return False
        return True
