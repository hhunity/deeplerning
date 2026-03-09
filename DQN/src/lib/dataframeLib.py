import datetime
import pandas as pd

import pandas as pd
import numpy as np

def make_ohlc_from_ticks(df: pd.DataFrame, interval: str = '1T') -> np.ndarray:
    """
    tickデータから指定時間足のOHLCを生成

    Parameters:
        df : pandas.DataFrame
            - 'time' : datetime列
            - 'bid'  : 価格列
        interval : str
            - リサンプリング単位（例: '1T'=1分, '5T'=5分, '1S'=1秒）

    Returns:
        np.ndarray : OHLCのNumPy配列（open, high, low, close）
    """
    # 時間列を datetime に変換（必要なら）
    df['time'] = pd.to_datetime(df['time'])

    # インデックスを時間に
    df = df.set_index('time')

    # bid列を使ってリサンプリングでOHLC作成
    ohlc = df['bid'].resample(interval).ohlc()

    # 欠損を除外（必要に応じて fillna や補完）
    ohlc = ohlc.dropna()

    return ohlc.to_numpy()