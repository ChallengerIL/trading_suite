from df_parser import Parser
from metatrader import mt5
from config import SYMBOL_HEAD, SYMBOL_TAIL, DEVIATION


class Currency:

    def __init__(self, pair, strategy, params, logger):
        self.name = SYMBOL_HEAD + pair + SYMBOL_TAIL
        self.params = params
        self.df = Parser(pair=pair, strategy=strategy, start=0, end=1, trading=True, rsi_vwap_period=self.params[4]).m1
        self.point = mt5.symbol_info(self.name).point
        self.deviation = DEVIATION
        self.logger = logger
        self.open_positions = mt5.positions_get(symbol=self.name)
        self.profits = []
        self.losses = []
        self.bought_idx = []
        self.sold_idx = []
        self.closed_buy_idx = []
        self.closed_sell_idx = []
        self.pips = 0
        self.profit_trades = 0
        self.loss_trades = 0
        self.profit_longs = 0
        self.loss_longs = 0
        self.profit_shorts = 0
        self.loss_shorts = 0
        self.total_trades = 0
        self.index = self.df.index

    def m15_rsi_vwap(self, account, index, lot_size, equity):
        # if len(self.open_positions) > 0:
        #     for i in range(len(self.open_positions)):
        #         if self.open_positions[i].type == 0:
        #             if self.rsi_vwap[index - 1] > self.params[6]:
        #                 position_id = self.open_positions[i].ticket
        #                 price = mt5.symbol_info_tick(self.name).bid
        #                 lot_size = self.open_positions[i].volume
        #                 account.close_order(
        #                 direction="long", strategy="rsi_vwap", symbol=self.name, position_id=position_id, price=price,
        #                 lot_size=lot_size, deviation=self.deviation, currency_logger=self.logger
        #                 )
        #         elif self.open_positions[i].type == 1:
        #             if self.rsi_vwap[index - 1] < self.params[7]:
        #                 position_id = self.open_positions[i].ticket
        #                 price = mt5.symbol_info_tick(self.name).ask
        #                 lot_size = self.open_positions[i].volume
        #                 account.close_order(
        #                 direction="short", strategy="rsi_vwap", symbol=self.name, position_id=position_id, price=price,
        #                 lot_size=lot_size, deviation=self.deviation, currency_logger=self.logger
        #                 )

        if round(self.df.rsi_vwap[index-2], 2) <= self.params[2] < round(self.df.rsi_vwap[index-1], 2):
            price = mt5.symbol_info_tick(self.name).ask
            margin = self.check_margin(mt5.ORDER_TYPE_BUY, price, lot_size)

            if margin <= equity:
                account.place_order(direction="long", symbol=self.name, price=price, deviation=self.deviation,
                                    currency_logger=self.logger, tp=self.params[0], sl=self.params[1], point=self.point)

        if round(self.df.rsi_vwap[index-2], 2) >= self.params[3] > round(self.df.rsi_vwap[index-1], 2):
            price = mt5.symbol_info_tick(self.name).bid
            margin = self.check_margin(mt5.ORDER_TYPE_SELL, price, lot_size)

            if margin <= equity:
                account.place_order(direction="short", symbol=self.name, price=price, deviation=self.deviation,
                                    currency_logger=self.logger, tp=self.params[0], sl=self.params[1], point=self.point)

    def check_margin(self, action, price, lot_size):
        margin = mt5.order_calc_margin(action, self.name, lot_size, price)
        return margin
