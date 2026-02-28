from get_data_yfinance import get_data
import pandas as pd


symbols = ['^VIX']
start = 1070236800
end = 1748904912

for symbol in symbols:
    df = get_data(start, end, symbol)
    df.to_csv(f'./data/{symbol}.csv')

# gbp_usd = pd.read_csv('./data/FXB.csv', index_col='Date')
# usd_jpy = pd.read_csv('./data/FXY.csv', index_col='Date')
# tqqq = pd.read_csv('./data/TQQQ.csv', index_col='Date')
# gld = pd.read_csv('./data/GLD.csv', index_col='Date')
# spx = pd.read_csv('./data/^GSPC.csv', index_col='Date')
# dix_gex = pd.read_csv('./data/DIX.csv', index_col='Date')
# vix = pd.read_csv('./data/vix.csv', index_col='Date')
# reverse_repo = pd.read_csv('./data/fed-reverse-repo.csv', index_col='Date')

# #trade on open
# #we will crate new bars (today's open, yesterday's high, yesterday's low, yesterday's close, 
# #                        yesterday's volume, yesterday's tickvol, yesterday's SHROUT)
# mask = tqqq.columns != 'Open'
# tqqq.loc[:, mask] = tqqq.loc[:, mask].shift(1)

# mask = gbp_usd.columns != 'Open'
# gbp_usd.loc[:, mask] = gbp_usd.loc[:, mask].shift(1)
# usd_jpy.loc[:, mask] = usd_jpy.loc[:, mask].shift(1)
# gld.loc[:, mask] = gld.loc[:, mask].shift(1)
# spx.loc[:, mask] = spx.loc[:, mask].shift(1)

# dix_gex['dix'] = dix_gex['dix'].shift(1)
# dix_gex['gex'] = dix_gex['gex'].shift(1)
# vix['Close'] = vix['Close'].shift(1)
# reverse_repo['RRPONTSYD'] = reverse_repo['RRPONTSYD'].shift(1)

# #covert the date index to datetime object
# dix_gex.index = pd.to_datetime(dix_gex.index)
# tqqq.index = pd.to_datetime(tqqq.index)
# gbp_usd.index = pd.to_datetime(gbp_usd.index)
# usd_jpy.index = pd.to_datetime(usd_jpy.index)
# vix.index = pd.to_datetime(vix.index)
# reverse_repo.index = pd.to_datetime(reverse_repo.index, dayfirst=True)
# gld.index = pd.to_datetime(gld.index)
# spx.index = pd.to_datetime(spx.index)


# df = (
#     tqqq
#     .join(dix_gex)
#     .join(gbp_usd, rsuffix='_GBPUSD')
#     .join(usd_jpy, rsuffix='_USDJPY')
#     .join(vix, rsuffix='_VIX')
#     .join(reverse_repo)
#     .join(gld, rsuffix='_GLD')
#     .join(spx, rsuffix='_SPX')
# )

# d = create_features(df)

# d.drop(['SHROUT', 'Low', 'High', 'Close', 'Volume', 'Open', 'Tickvol',
#         'Low_GBPUSD', 'High_GBPUSD', 'Close_GBPUSD', 'Volume_GBPUSD', 'Open_GBPUSD', 'Tickvol_GBPUSD', 'SHROUT_GBPUSD',
#         'Low_USDJPY', 'High_USDJPY', 'Close_USDJPY', 'Volume_USDJPY', 'Open_USDJPY', 'Tickvol_USDJPY', 'SHROUT_USDJPY',
#         'Low_VIX', 'High_VIX', 'Close_VIX', 'Open_VIX',
#         'Low_GLD', 'High_GLD', 'Close_GLD', 'Volume_GLD', 'Open_GLD', 'Tickvol_GLD', 'SHROUT_GLD',
#         'Low_SPX', 'High_SPX', 'Close_SPX', 'Volume_SPX', 'Open_SPX', 'Tickvol_SPX', 'SHROUT_SPX',
#         'returns'
#         ], axis=1, inplace=True)

# d.dropna(subset=['dix', 'gex', 'fwd_return_1', 'fwd_return_1_label'], inplace=True)


# d.to_csv('./data/processed_data1.csv')





