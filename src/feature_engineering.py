import gc
import pandas as pd

'''
These functions are used for the feature engineering at the OTTO competition.
You should prepare
- input dataframe with the candidates
- session feature dataframe
- aid feature dataframe
in advance.
'''


# reduce memory by converting type of the dataframe
def reduce_memory(df):
    df['session'] = df['session'].astype('int32')
    df['aid'] = df['aid'].astype('int32')
    df['score_click'] = df['score_click'].astype('float32')
    df['score_cart'] = df['score_cart'].astype('float32')
    df['score_buy'] = df['score_buy'].astype('float32')
    df['score_click_only'] = df['score_click_only'].astype('float32')
    df['score_cart_only'] = df['score_cart_only'].astype('float32')
    df['score_buy_only'] = df['score_buy_only'].astype('float32')
    df['session_action_count'] = df['session_action_count'].astype('int16')

    click_topn_list = [5, 10, 20, 30]
    cart_topn_list = [5, 15, 20]
    buy_topn_list = [5, 15, 20]

    for i in click_topn_list:
        df[f'n_clicks_{i}'] = df[f'n_clicks_{i}'].astype('int8')
    for i, val in enumerate(cart_topn_list):
        for j in range(2):
            df[f'n_carts_{val}_ver{j}'] = df[f'n_carts_{val}_ver{j}'].astype('int8')
    for i, val in enumerate(buy_topn_list):
        for j in range(2):
            df[f'n_buys_{val}_ver{j}'] = df[f'n_buys_{val}_ver{j}'].astype('int8')
    return df


# join session features to the input dataframe
def join_session_features(df, path):
    session_df = pd.read_parquet(path)
    session_df['session'] = session_df['session'].astype('int32')
    session_df['session_action_count'] = session_df['session_action_count'].astype('int16')
    session_df['session_click_count'] = session_df['session_click_count'].astype('int16')
    session_df['session_cart_count'] = session_df['session_cart_count'].astype('int16')
    session_df['session_order_count'] = session_df['session_order_count'].astype('int16')
    session_df['session_type_mean'] = session_df['session_type_mean'].astype('float32')
    session_df['session_click_rate'] = session_df['session_click_rate'].astype('float32')
    session_df['session_cart_rate'] = session_df['session_cart_rate'].astype('float32')
    session_df['session_order_rate'] = session_df['session_order_rate'].astype('float32')
    session_df['session_last_type'] = session_df['session_last_type'].astype('int8')
    session_df['session_first_action_ts_diff'] = session_df['session_first_action_ts_diff'].astype('int32')
    session_df['session_last_action_ts_diff'] = session_df['session_last_action_ts_diff'].astype('int32')
    session_df['session_ts_period'] = session_df['session_ts_period'].astype('int32')
    session_df['session_mean_action_ts_diff'] = session_df['session_mean_action_ts_diff'].astype('float32')
    session_df['session_unique_aid_action_count'] = session_df['session_unique_aid_action_count'].astype('int16')
    session_df['session_unique_aid_click_count'] = session_df['session_unique_aid_click_count'].astype('int16')
    session_df['session_unique_aid_cart_count'] = session_df['session_unique_aid_cart_count'].astype('int16')
    session_df['session_unique_aid_order_count'] = session_df['session_unique_aid_order_count'].astype('int16')
    session_df['session_click_rate'] = session_df['session_click_rate'].astype('float32')
    session_df['session_cart_rate'] = session_df['session_cart_rate'].astype('float32')
    session_df['session_order_rate'] = session_df['session_order_rate'].astype('float32')
    session_df['session_unique_aid_click_rate'] = session_df['session_unique_aid_click_rate'].astype('float32')
    session_df['session_unique_aid_cart_rate'] = session_df['session_unique_aid_cart_rate'].astype('float32')
    session_df['session_unique_aid_order_rate'] = session_df['session_unique_aid_order_rate'].astype('float32')
    session_df['session_total_uu_action_ratio'] = session_df['session_total_uu_action_ratio'].astype('float32')
    session_df['session_clicks_uu_action_ratio'] = session_df['session_clicks_uu_action_ratio'].astype('float32')
    session_df['session_carts_uu_action_ratio'] = session_df['session_carts_uu_action_ratio'].astype('float32')
    session_df['session_orders_uu_action_ratio'] = session_df['session_orders_uu_action_ratio'].astype('float32')

    week_list = ['4weeks', '2weeks', '1week']
    for i in week_list:
        lis = [f'aid_clicks_count_{i}',
               f'aid_carts_count_{i}',
               f'aid_orders_count_{i}',
               f'aid_total_count_{i}',
               f'aid_total_uu_{i}',
               f'aid_clicks_uu_{i}',
               f'aid_carts_uu_{i}',
               f'aid_orders_uu_{i}',
               f'aid_total_uu_action_ratio_{i}',
               f'aid_clicks_uu_action_ratio_{i}',
               f'aid_carts_uu_action_ratio_{i}',
               f'aid_orders_uu_action_ratio_{i}']
        for l in lis:
            session_df[f'session_mean_{l}'] = session_df[f'session_mean_{l}'].astype('float32')

    remove_col = ['session_action_count']

    df = df.drop(remove_col, axis=1)
    df = df.merge(session_df, 'left', 'session')
    del session_df
    gc.collect()

    return df


# join aid features to the input dataframe
def join_aid_features(df, path):
    aid_df = pd.read_parquet(path)
    week_list = ['4weeks', '2weeks', '1week']
    aid_df['aid'] = aid_df['aid'].astype('int32')
    for i in week_list:
        aid_df[f'aid_total_count_{i}'] = aid_df[f'aid_total_count_{i}'].astype('int32')
        aid_df[f'aid_clicks_count_{i}'] = aid_df[f'aid_clicks_count_{i}'].astype('int32')
        aid_df[f'aid_carts_count_{i}'] = aid_df[f'aid_carts_count_{i}'].astype('int16')
        aid_df[f'aid_orders_count_{i}'] = aid_df[f'aid_orders_count_{i}'].astype('int16')
        aid_df[f'aid_clicks_rank_{i}'] = aid_df[f'aid_clicks_rank_{i}'].astype('int32')
        aid_df[f'aid_carts_rank_{i}'] = aid_df[f'aid_carts_rank_{i}'].astype('int32')
        aid_df[f'aid_orders_rank_{i}'] = aid_df[f'aid_orders_rank_{i}'].astype('int32')

        aid_df[f'aid_total_uu_{i}'] = aid_df[f'aid_total_uu_{i}'].astype('int32')
        aid_df[f'aid_clicks_uu_{i}'] = aid_df[f'aid_clicks_uu_{i}'].astype('int32')
        aid_df[f'aid_carts_uu_{i}'] = aid_df[f'aid_carts_uu_{i}'].astype('int32')
        aid_df[f'aid_orders_uu_{i}'] = aid_df[f'aid_orders_uu_{i}'].astype('int32')
        aid_df[f'aid_total_uu_action_ratio_{i}'] = aid_df[f'aid_total_uu_action_ratio_{i}'].astype('float32')
        aid_df[f'aid_clicks_uu_action_ratio_{i}'] = aid_df[f'aid_clicks_uu_action_ratio_{i}'].astype('float32')
        aid_df[f'aid_carts_uu_action_ratio_{i}'] = aid_df[f'aid_carts_uu_action_ratio_{i}'].astype('float32')
        aid_df[f'aid_orders_uu_action_ratio_{i}'] = aid_df[f'aid_orders_uu_action_ratio_{i}'].astype('float32')
        aid_df[f'aid_mean_session_action_count_{i}'] = aid_df[f'aid_mean_session_action_count_{i}'].astype('float32')
        aid_df[f'aid_mean_session_click_count_{i}'] = aid_df[f'aid_mean_session_click_count_{i}'].astype('float32')
        aid_df[f'aid_mean_session_cart_count_{i}'] = aid_df[f'aid_mean_session_cart_count_{i}'].astype('float32')
        aid_df[f'aid_mean_session_order_count_{i}'] = aid_df[f'aid_mean_session_order_count_{i}'].astype('float32')
        aid_df[f'aid_mean_session_type_mean_{i}'] = aid_df[f'aid_mean_session_type_mean_{i}'].astype('float32')

    for j in ['clicks', 'carts', 'orders']:
        for k in [2, 4]:
            aid_df[f'aid_{j}_count_rate_1_{k}'] = aid_df[f'aid_{j}_count_rate_1_{k}'].astype('float32')
            aid_df[f'aid_{j}_uu_rate_1_{k}'] = aid_df[f'aid_{j}_uu_rate_1_{k}'].astype('float32')
        # only read at the first j loop
        if j == 'clicks':
            aid_df[f'aid_total_uu_rate_1_{k}'] = aid_df[f'aid_total_uu_rate_1_{k}'].astype('float32')

    df = df.merge(aid_df, 'left', 'aid')
    del aid_df
    gc.collect()
    return df


# generate interactive features from the session and aid features
def join_interactive_features(df):
    # session base interactive features
    week_list = ['4weeks']

    for i in week_list:
        # ratio feature
        lis1 = [f'aid_clicks_count_{i}',
                f'aid_carts_count_{i}',
                f'aid_orders_count_{i}',
                f'aid_total_count_{i}',
                f'aid_total_uu_{i}',
                f'aid_clicks_uu_{i}',
                f'aid_carts_uu_{i}',
                f'aid_orders_uu_{i}', ]
        for l1 in lis1:
            df[f'i_ratio_{l1}'] = df[f'session_mean_{l1}'] / (df[l1] + 0.000001)
            df[f'i_ratio_{l1}'] = df[f'i_ratio_{l1}'].astype('float32')

        # aid based features
        lis3 = ['session_action_count',
                'session_click_count',
                'session_cart_count',
                'session_order_count', ]

        for l3 in lis3:
            df[f'i_ratio_{l3}_{i}'] = df[f'aid_mean_{l3}_{i}'] / (df[l3] + 0.000001)
            df[f'i_ratio_{l3}_{i}'] = df[f'i_ratio_{l3}_{i}'].astype('float32')
        df[f'i_diff_session_type_mean_{i}'] = df[f'aid_mean_session_type_mean_{i}'] - df['session_type_mean']
    return df
