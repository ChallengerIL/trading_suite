from metatrader import mt5, connect, shutdown
import h5py
from math import ceil
import numpy as np
from config import SYMBOL_HEAD, SYMBOL_TAIL, FILES_DIR, BUFFER_BARS
import os


TFS_DICT = {
        "M1": mt5.TIMEFRAME_M1, "M2": mt5.TIMEFRAME_M2, "M3": mt5.TIMEFRAME_M3, "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5, "M6": mt5.TIMEFRAME_M6, "M10": mt5.TIMEFRAME_M10, "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15, "M20": mt5.TIMEFRAME_M20, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2, "H3": mt5.TIMEFRAME_H3, "H4": mt5.TIMEFRAME_H4, "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8, "H12": mt5.TIMEFRAME_H12, "D1": mt5.TIMEFRAME_D1, "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }


def view_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to keep.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    formats = [dt.fields[name][0] for name in names]
    offsets = [dt.fields[name][1] for name in names]
    itemsize = a.dtype.itemsize
    newdt = np.dtype(dict(names=names,
                          formats=formats,
                          offsets=offsets,
                          itemsize=itemsize))
    b = a.view(newdt)
    return b


def remove_fields(a, names):
    """
    `a` must be a numpy structured array.
    `names` is the collection of field names to remove.

    Returns a view of the array `a` (not a copy).
    """
    dt = a.dtype
    keep_names = [name for name in dt.names if name not in names]
    return view_fields(a, keep_names)


def prepare(df):
    if df is None:
        print("Parsing failed")
        print("Check the Internet Connection or the Symbol's name")
        print(mt5.last_error())
        quit()

    df.dtype.names = 'time', 'open', 'high', 'low', 'close', 'vol', 'spread', 'empty'
    df = remove_fields(df, 'empty')

    return df


class Parser:

    TF_MULTIPLIERS = {
        "M1": 60, "M5": 12, "M15": 4, "M30": 2, "H1": 1, "H4": 0.25, "D1": 0.042, "W1": 0.006, "M": 0.0014,
    }

    def __init__(self, pair, strategy, start, end, **kw):
        if "trading" not in kw:
            connect()

        self.name = SYMBOL_HEAD + pair + SYMBOL_TAIL
        self.symbol_info = mt5.symbol_info(self.name)
        all_tfs = list()
        for k, v in strategy['indicators'].items():
            all_tfs += list(v.keys())

        self.tfs = sorted(list(set(all_tfs)), key=strategy['tfs'].index)

        if not self.symbol_info.visible:
            if not mt5.symbol_select(self.name, True):
                print("symbol_select({}}) failed", self.name)
                quit()

        if "trading" in kw:
            self.df = {
                tf: prepare(df=mt5.copy_rates_from_pos(self.name, TFS_DICT[tf], start, ceil(
                    (end + 1) * self.TF_MULTIPLIERS[tf]) + BUFFER_BARS)) for tf in self.tfs
            }
        else:
            self.df = {
                tf: prepare(df=mt5.copy_rates_range(self.name, TFS_DICT[tf], start, end)) for tf in self.tfs
            }
            if "save" in kw:
                if os.path.exists(f"{FILES_DIR}pairs_data.h5") and pair == strategy["pairs"][0]:
                    os.remove(f"{FILES_DIR}pairs_data.h5")

                with h5py.File(f"{FILES_DIR}pairs_data.h5", 'a') as file:
                    [file.create_dataset(f"{tf}_{pair}", data=self.df[tf], dtype=self.df[tf].dtype) for tf in self.df]

            shutdown()
