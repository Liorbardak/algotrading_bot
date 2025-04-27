import pandas as pd
import numpy as np
import pkgutil
import inspect
import importlib
import argparse
from typing import List, Dict, Tuple
import  bots as algotrading_bots
from sim import run_trade_sim

def get_bots_to_run(bots_str: str):
    """ get bots to sim on """
    bots_names = bots_str.split(',')
    bots_to_run = []

    for name in  dir(algotrading_bots):
        if inspect.isclass(getattr(algotrading_bots, name)):
            bot = getattr(algotrading_bots, name)()
            if hasattr(bot,'_name')  and bot._name in bots_names:
                bots_to_run.append(bot)
    if (len(bots_to_run) == 0):
        print(f"{bots_str}  does not contain valid bots ")
    return bots_to_run


def main():
    parser = argparse.ArgumentParser(description='Run algo trading sim')
    parser.add_argument('-i', type=str, default= "C:/Users/dadab/projects/algotrading/data/snp500", help='input dir ')
    parser.add_argument('-o', type=str, default= "C:/Users/dadab/projects/algotrading/results/playground", help='out dir')
    parser.add_argument('-b', type=str, default="charnybotv1,charnybotBase", help='list of bots to run ')

    args = parser.parse_args()
    trade_bots = get_bots_to_run(args.b)
    run_trade_sim(datadir=args.i, results_dir=args.o, trade_bots=trade_bots)

if __name__ == "__main__":
    main()