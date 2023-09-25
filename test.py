from tester import backtester
from df_parser import Parser
from metatrader import connect, shutdown
from config import STRATEGIES, TIMEZONE, START, END
import time as t
from datetime import datetime
import pytz
import numpy as np
import os
import math

# os.system(f"python trainer.py GBPUSD example_strategy")
# quit()

# exec(open('ga.py').read())

# timezone = pytz.timezone('Asia/Tel_Aviv')
# # today = datetime.now(tz=timezone)
# today = datetime.now().replace(tzinfo=timezone)
# print(today)

# start = datetime(2020, 1, 10, tzinfo=TIMEZONE)
# end = datetime(2022, 5, 11, hour=13, tzinfo=TIMEZONE)

# 200 days of training
# pairs = ["GBPUSD"]
# params = [[41, -15, 11, 5, -0.298276, 0.342503, -0.060626, 0.379231, -0.151108, -0.482984, 0.408558, 11, 4.198,
# -2.544, 0.778563, 11, 0, 12, 8, 18, 0.798253, -0.775652, -0.610665, 0.150835, 0.333732, 0.657332]]

# 50 days, 419.2 pips, 10.23% drawdown
# pairs = ["GBPUSD"]
# params = [[24, -10, 100, 9, -0.740755, -0.279031, -0.249923, 0.701846, 0.85625, 0.075903, 0.017171, 0.561, 0.697574,
#            -0.442744, 23, 0.154, 0.538, 0.280815, 4, 0, 0, 8, 9, 17]]

# pairs = ["GBPUSD", "EURUSD", "EURJPY", "USDCHF"]
# params = [[8.737919080411682, 21.045104575121513, 8.441006337722525]]
params = [
    [68.16267646268227, 29.551871547642865, -19.55960487332239, 9.576649210393155, 96.54911558058686, 6.0220227391377925, 13.076362728376695, 28.581955667351625], [100, 10], [100, 10], [100, 10], [100, 10], [100, 10], [100, 10]
]

# pairs = ["GBPUSD", "EURUSD", "EURJPY", "USDJPY", "USDCAD", "AUDUSD", "USDCHF"]
# params = [
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
#     [50, -30, 40, 100, 13, 22, 10, 26, 22, 10.34, 81.51, 66.04, 94.72, -48.78, -9.85, 9, 18],
# ]

# Concentration 30, 20

#
# # print(START)
# # print(END)
#
# connect()

pairs = STRATEGIES["example_strategy"]['pairs']

# Uncomment to update the source data (MT5 is required)
for pair in pairs:
    Parser(pair=pair, strategy=STRATEGIES["example_strategy"], save=True, start=START, end=END)

# shutdown()
# start = t.time()
backtester(strategy=STRATEGIES["example_strategy"], params=params, plotting=True, verbose=True, cushion=0)
# end = t.time()
# print("Elapsed (with compilation) = %s" % (end - start))

# connect()
# currency = Currency(pair=pairs[0], strategy=STRATEGIES["rsi_vwap"], params=params[0], start=START,
#                     end=END)
# shutdown()
# currency.plotter()

# start = t.time()
# update_params(pair, end_pos=end_pos, test=test)

# results = backtester(
#     pairs=pairs, strategy=STRATEGIES["rsi_vwap"], params=params, plotting=True, verbose=True, cushion=0
# )

# end = t.time()
# print("Elapsed (with compilation) = %s" % (end - start))
# # results = backtester(
# #     pairs=pairs, strategy=STRATEGIES["rsi_vwap"], params=params, plotting=True, verbose=True, cushion=0
# # )
# print(results)
# # quit()