import yfinance as yf
import pandas as pd
import numpy as np
import time
import datetime
import logging
import sys
from pathlib import Path
import subprocess

sys.path.append(str(Path(__file__).resolve().parent / "src"))
from stock_indicator import indicators

def screen(parameter, debug_parameter, symbols_dir: Path | str = Path("US-Stock-Symbols")):
    SCAN_TYPE = parameter[0]                # Scan marks for: K1 or FTD or or PBB or BuyRating or BOTH  ( K1 sell mark has bug, BOTH=K1+FTD )
    PERIOD = parameter[1]                   # scan for marks within n days
    INTERVAL = parameter[2]                 # Valid intervals: [1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo]
    PRICE_ABOVE = parameter[3]              # included PRICE_ABOVE
    VOLUMN_ABOVE = parameter[4]             # included VOLUMN_ABOVE, suggest 0 or 500000

    DEBUG = debug_parameter[0]              # debug flag
    DEBUG_SYMBOL = debug_parameter[1]       # debug symbol

    # Get the current date, then Format it as a string
    current_date = datetime.date.today()
    date_string = current_date.strftime('%Y-%m-%d')

    #--------- Logger config
    # Create a logger
    logger = logging.getLogger(__name__)
    # Set the level of the logger. This can be adjusted to restrict which logs are written.
    logger.setLevel(logging.INFO)

    # Create a file handler
    handler = logging.FileHandler(f'./log/log_{date_string}.txt')
    handler.setLevel(logging.INFO)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)
    #---------

    start_time = time.time()

    symbols_dir = Path(symbols_dir)
    if not symbols_dir.exists():
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/rreichel3/US-Stock-Symbols.git",
                str(symbols_dir),
            ],
            check=True,
        )
        if not symbols_dir.exists():
            raise FileNotFoundError(f"Failed to clone repository into {symbols_dir}")

    # Read JSON files
    amex_df = pd.read_json(symbols_dir / "amex" / "amex_tickers.json")
    nasdaq_df = pd.read_json(symbols_dir / "nasdaq" / "nasdaq_tickers.json")
    nyse_df = pd.read_json(symbols_dir / "nyse" / "nyse_tickers.json")

    # Concatenate DataFrames
    combined_df = pd.concat([amex_df, nasdaq_df, nyse_df], ignore_index=True)

    # convert to yf name
    combined_df[0] = combined_df[0].replace(r'[\/]', '-', regex=True)
    combined_df[0] = combined_df[0].replace(r'[\^]', '-P', regex=True)

    # Get unique values
    LIST_SYMBOL = combined_df[0].unique().tolist()
    number_symbol = len(LIST_SYMBOL)

    if DEBUG:
        if SCAN_TYPE == 'K1':
            indicators.K1(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'FTD':
            indicators.ftd(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'PBB':
            indicators.pbb(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'BuyRating':
            indicators.buyRating(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'vol_up':
            indicators.vol_up(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'mm_stage2':
            indicators.mm_stage2(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/{SCAN_TYPE}_"+DEBUG_SYMBOL+".csv", index=False, header=True)
        elif SCAN_TYPE == 'BOTH':
            indicators.K1(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/K1_"+DEBUG_SYMBOL+".csv", index=False, header=True)
            indicators.ftd(DEBUG_SYMBOL, 1, 0, 0, INTERVAL, DEBUG).to_csv(f"./stock_output/FTD_"+DEBUG_SYMBOL+".csv", index=False, header=True)
    else:
        buy_mark_list = []
        process_count = 0
        _rating = 0.0
        for target_symbol in LIST_SYMBOL:
            if SCAN_TYPE == 'K1':
                _isBuyMark = indicators.K1(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'FTD':
                _isBuyMark, _rating = indicators.ftd(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'PBB':
                _isBuyMark = indicators.pbb(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'BuyRating':
                _isBuyMark = indicators.buyRating(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'vol_up':
                _isBuyMark = indicators.vol_up(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'mm_stage2':
                _isBuyMark = indicators.mm_stage2(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
            elif SCAN_TYPE == 'BOTH':
                _K1_check = indicators.K1(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
                _FTD_check = indicators.pbb(target_symbol, PERIOD, PRICE_ABOVE, VOLUMN_ABOVE, INTERVAL, DEBUG)
                _isBuyMark = _K1_check & _FTD_check
            if _isBuyMark:
                if _rating != 0.0:
                    buy_mark_list.append(target_symbol + "," + str(_rating))
                else:
                    buy_mark_list.append(target_symbol)
            process_count += 1
            message = f"processing progress: {process_count}/{number_symbol}"
            print(message)
            logger.info(message)  # log the message

        df_buy_mark = pd.DataFrame(buy_mark_list)
        if not df_buy_mark.empty:
        # convert to futu name
            df_buy_mark[0] = df_buy_mark[0].replace(r'[\/]', '-', regex=True)
            df_buy_mark[0] = df_buy_mark[0].replace(r'[\^]', '-', regex=True)
        df_buy_mark.to_csv(f"./result_output/watch_list_{date_string}_{SCAN_TYPE}.txt", index=False, header=False)

    end_time = time.time()
    print('full list completed: ', end_time - start_time)
