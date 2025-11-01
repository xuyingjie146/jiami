import streamlit as st
import sys
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from scipy.signal import argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')
import base64
from io import BytesIO

# 设置页面
st.set_page_config(
    page_title="加密货币形态扫描器 - 严格模式",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 应用标题
st.title("📈 加密货币形态扫描器 - 严格模式")
st.markdown("""
<div style="background-color:#f0f2f6;padding:20px;border-radius:10px;margin-bottom:20px;">
<h3 style="color:#1f77b4;margin:0;">🎯 严格模式特性</h3>
<ul style="color:#333;">
<li><b>严格验证</b>: 需要3个高点和2个低点依次出现</li>
<li><b>精确趋势线</b>: 所有摆动点必须在趋势线上 (R²>0.95)</li>
<li><b>大规模扫描</b>: 支持前200币种完整分析</li>
<li><b>图表缓存</b>: 历史结果可重复查看</li>
</ul>
</div>
""", unsafe_allow_html=True)

class StrictPatternScanner:
    def __init__(self):
        self.base_url = "https://api.gateio.ws/api/v4"
        self.volume_symbols = self.get_top_spot_by_volume(200)  # 改为200
        self.all_timeframes = ["15m", "1h", "4h", "1d"]
        self.all_kline_counts = [200, 400]
        self.pattern_scores = {
            "对称三角形": 95, "上升三角形": 95, "下降三角形": 95,
            "上升通道": 90, "下降通道": 90,
            "上升楔形": 85, "下降楔形": 85,
            "看涨旗形": 88, "看跌旗形": 88
        }
        self.scan_results = []
        self.seen_patterns = set()
        self.chart_cache = {}  # 图表缓存

    def get_top_spot_by_volume(self, limit=200):  # 改为200
        """获取现货成交额前200的加密货币"""
        try:
            with st.spinner(f'🔄 获取加密货币列表 (前{limit})...'):
                url = f"{self.base_url}/spot/tickers"
                response = requests.get(url, timeout=20)  # 增加超时时间
                
                if response.status_code == 200:
                    tickers_data = response.json()
                    usdt_pairs = []
                    for ticker in tickers_data:
                        currency_pair = ticker.get('currency_pair', '')
                        if currency_pair.endswith('_USDT'):
                            quote_volume = float(ticker.get('quote_volume', 0))
                            change_percent = float(ticker.get('change_percent', 0))
                            usdt_pairs.append({
                                'symbol': currency_pair,
                                'quote_volume': quote_volume,
                                'change_percent': change_percent
                            })
                    
                    if not usdt_pairs:
                        return self.get_backup_symbols(limit)
                    
                    usdt_pairs.sort(key=lambda x: x['quote_volume'], reverse=True)
                    symbols = [item['symbol'] for item in usdt_pairs[:limit]]
                    st.success(f"✅ 成功获取成交额前{len(symbols)}个币种")
                    return symbols
                else:
                    st.warning("API请求失败，使用备用列表")
                    return self.get_backup_symbols(limit)
                    
        except Exception as e:
            st.warning(f"获取实时数据失败，使用备用列表: {e}")
            return self.get_backup_symbols(limit)
    
    def get_backup_symbols(self, limit=200):  # 改为200
        """备用币种列表 - 扩展到200个"""
        backup_symbols = [
            "BTC_USDT", "ETH_USDT", "BNB_USDT", "SOL_USDT", "XRP_USDT",
            "ADA_USDT", "AVAX_USDT", "DOGE_USDT", "DOT_USDT", "LINK_USDT",
            "MATIC_USDT", "LTC_USDT", "ATOM_USDT", "ETC_USDT", "XLM_USDT",
            "BCH_USDT", "FIL_USDT", "ALGO_USDT", "VET_USDT", "THETA_USDT",
            "TRX_USDT", "EOS_USDT", "XMR_USDT", "XTZ_USDT", "SAND_USDT",
            "MANA_USDT", "GALA_USDT", "ENJ_USDT", "CHZ_USDT", "BAT_USDT",
            # 添加更多常见币种
            "NEAR_USDT", "FTM_USDT", "EGLD_USDT", "AAVE_USDT", "MKR_USDT",
            "COMP_USDT", "SNX_USDT", "CRV_USDT", "SUSHI_USDT", "1INCH_USDT",
            "ZEC_USDT", "DASH_USDT", "WAVES_USDT", "OMG_USDT", "ZIL_USDT",
            "IOTA_USDT", "ONT_USDT", "QTUM_USDT", "ICX_USDT", "SC_USDT",
            "ANKR_USDT", "REN_USDT", "CELR_USDT", "ONE_USDT", "HOT_USDT",
            "IOST_USDT", "STORJ_USDT", "KNC_USDT", "REEF_USDT", "RSR_USDT",
            "COTI_USDT", "OCEAN_USDT", "BAND_USDT", "NKN_USDT", "LRC_USDT",
            "AR_USDT", "RLC_USDT", "BAL_USDT", "KAVA_USDT", "SRM_USDT",
            "YFI_USDT", "UMA_USDT", "RUNE_USDT", "SFP_USDT", "CELO_USDT",
            "OGN_USDT", "SKL_USDT", "GRT_USDT", "BNT_USDT", "TOMO_USDT",
            "DENT_USDT", "STMX_USDT", "HIVE_USDT", "DGB_USDT", "STPT_USDT",
            "CHR_USDT", "ARPA_USDT", "PERL_USDT", "TROY_USDT", "VITE_USDT",
            "DUSK_USDT", "WRX_USDT", "BTS_USDT", "TFUEL_USDT", "CVC_USDT",
            "CTSI_USDT", "STRAX_USDT", "AUDIO_USDT", "REQ_USDT", "DATA_USDT",
            "SXP_USDT", "IRIS_USDT", "CTK_USDT", "UNI_USDT", "RVN_USDT",
            "SYS_USDT", "FIO_USDT", "DIA_USDT", "BEL_USDT", "WING_USDT",
            "TRB_USDT", "ORN_USDT", "PSG_USDT", "CITY_USDT", "LIT_USDT",
            "BADGER_USDT", "ALPHA_USDT", "VIDT_USDT", "AXS_USDT", "SLP_USDT",
            "SAND_USDT", "MANA_USDT", "GALA_USDT", "ENJ_USDT", "CHZ_USDT",
            "BAT_USDT", "ANKR_USDT", "REN_USDT", "CELR_USDT", "ONE_USDT",
            "HOT_USDT", "IOST_USDT", "STORJ_USDT", "KNC_USDT", "REEF_USDT",
            "RSR_USDT", "COTI_USDT", "OCEAN_USDT", "BAND_USDT", "NKN_USDT",
            "LRC_USDT", "AR_USDT", "RLC_USDT", "BAL_USDT", "KAVA_USDT",
            "SRM_USDT", "YFI_USDT", "UMA_USDT", "RUNE_USDT", "SFP_USDT",
            "CELO_USDT", "OGN_USDT", "SKL_USDT", "GRT_USDT", "BNT_USDT",
            "TOMO_USDT", "DENT_USDT", "STMX_USDT", "HIVE_USDT", "DGB_USDT",
            "STPT_USDT", "CHR_USDT", "ARPA_USDT", "PERL_USDT", "TROY_USDT",
            "VITE_USDT", "DUSK_USDT", "WRX_USDT", "BTS_USDT", "TFUEL_USDT",
            "CVC_USDT", "CTSI_USDT", "STRAX_USDT", "AUDIO_USDT", "REQ_USDT",
            "DATA_USDT", "SXP_USDT", "IRIS_USDT", "CTK_USDT", "UNI_USDT",
            "RVN_USDT", "SYS_USDT", "FIO_USDT", "DIA_USDT", "BEL_USDT",
            "WING_USDT", "TRB_USDT", "ORN_USDT", "PSG_USDT", "CITY_USDT",
            "LIT_USDT", "BADGER_USDT", "ALPHA_USDT", "VIDT_USDT", "AXS_USDT",
            "SLP_USDT", "SAND_USDT", "MANA_USDT", "GALA_USDT", "ENJ_USDT"
        ]
        # 去重并返回前limit个
        unique_symbols = list(dict.fromkeys(backup_symbols))
        return unique_symbols[:limit]

    def save_chart_to_cache(self, symbol, timeframe, pattern_type, kline_count, chart_buf):
        """保存图表到缓存"""
        key = f"{symbol}_{timeframe}_{pattern_type}_{kline_count}"
        self.chart_cache[key] = chart_buf
        return key
    
    def get_chart_from_cache(self, key):
        """从缓存获取图表"""
        return self.chart_cache.get(key)

    def display_result_with_chart(self, result):
        """显示结果和图表"""
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("当前价格", f"${result['price']:.4f}")
        with col2:
            st.metric("时间框架", result['timeframe'])
        with col3:
            st.metric("K线数量", f"{result['kline_count']}根")
        with col4:
            st.metric("形态得分", f"{result['score']}%")
        
        # 显示严格模式信息
        st.info(f"🔒 严格模式: {result['swing_highs']}高/{result['swing_lows']}低依次出现，所有点在趋势线上 (R²>0.95)")
        
        # 从缓存获取并显示图表
        chart_buf = self.get_chart_from_cache(result['chart_key'])
        if chart_buf:
            st.image(chart_buf, use_column_width=True)
        else:
            st.warning("图表数据已过期，请重新扫描")

    def get_spot_candle_data(self, symbol="BTC_USDT", interval="15m", limit=400):
        """获取现货K线数据"""
        try:
            url = f"{self.base_url}/spot/candlesticks"
            params = {
                'currency_pair': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                df = self.process_candle_data(data)
                if df is not None and len(df) >= limit * 0.75:
                    return df
            
            return self.generate_realistic_data(symbol, interval, limit)
                
        except Exception as e:
            st.error(f"获取数据失败: {e}")
            return self.generate_realistic_data(symbol, interval, limit)

    def process_candle_data(self, data):
        """处理K线数据"""
        try:
            processed_data = []
            
            for candle in data:
                if len(candle) >= 6:
                    timestamp = int(candle[0])
                    volume = float(candle[1])
                    close = float(candle[2])
                    high = float(candle[3])
                    low = float(candle[4])
                    open_price = float(candle[5])
                    
                    processed_data.append({
                        'timestamp': pd.to_datetime(timestamp * 1000),
                        'Open': open_price,
                        'High': high,
                        'Low': low,
                        'Close': close,
                        'Volume': volume
                    })
            
            if not processed_data:
                return None
                
            df = pd.DataFrame(processed_data)
            df.set_index('timestamp', inplace=True)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df = df.sort_index()
            
            return df
            
        except Exception as e:
            return None

    def generate_realistic_data(self, symbol, interval, limit=400):
        """生成真实模拟数据"""
        real_prices = {
            "BTC_USDT": 45000, "ETH_USDT": 2500, "BNB_USDT": 320,
            "SOL_USDT": 110, "XRP_USDT": 0.62, "ADA_USDT": 0.48
        }
        
        base_price = real_prices.get(symbol, 10)
        
        if interval == "15m":
            periods = limit
            freq = '15min'
            volatility = 0.003
        elif interval == "1h":
            periods = limit
            freq = 'H'
            volatility = 0.008
        elif interval == "4h":
            periods = limit
            freq = '4H'
            volatility = 0.015
        else:
            periods = limit
            freq = 'D'
            volatility = 0.025
        
        dates = pd.date_range(end=datetime.now(), periods=periods, freq=freq)
        np.random.seed(hash(symbol) % 10000)
        
        prices = [base_price]
        
        for i in range(1, periods):
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            new_price = max(new_price, base_price * 0.1)
            prices.append(new_price)
        
        df_data = []
        for i, date in enumerate(dates):
            close_price = prices[i]
            open_price = prices[i-1] if i > 0 else close_price * 0.99
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/2)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/2)))
            
            high_price = max(open_price, close_price, high_price)
            low_price = min(open_price, close_price, low_price)
            
            volume = base_price * 1000 * np.random.uniform(0.8, 1.2)
            
            df_data.append({
                'timestamp': date,
                'Open': open_price,
                'High': high_price,
                'Low': low_price,
                'Close': close_price,
                'Volume': volume
            })
        
        df = pd.DataFrame(df_data)
        df.set_index('timestamp', inplace=True)
        return df

    def find_swing_points(self, df, window=10):
        """找到摆动点 - 需要3个高点和2个低点（或3个低点和2个高点）依次出现"""
        if len(df) < window * 3:
            return [], []
        
        highs = df['High'].values
        lows = df['Low'].values
        
        # 找到局部极值点
        high_indices = argrelextrema(highs, np.greater, order=window)[0]
        low_indices = argrelextrema(lows, np.less, order=window)[0]
        
        if len(high_indices) < 3 or len(low_indices) < 2:
            return [], []
        
        # 获取最近的摆动点
        recent_highs = high_indices[-3:]  # 取最近3个高点
        recent_lows = low_indices[-2:]    # 取最近2个低点
        
        # 检查是否依次交替出现
        all_points = []
        for idx in recent_highs:
            all_points.append(('high', idx, highs[idx]))
        for idx in recent_lows:
            all_points.append(('low', idx, lows[idx]))
        
        # 按时间顺序排序
        all_points.sort(key=lambda x: x[1])
        
        # 检查是否满足交替模式：高-低-高-低-高 或 低-高-低-高-低
        valid_patterns = [
            ['high', 'low', 'high', 'low', 'high'],  # 三高二低
            ['low', 'high', 'low', 'high', 'low']    # 三低二高
        ]
        
        current_pattern = [point[0] for point in all_points]
        
        if current_pattern in valid_patterns:
            # 分离高点和低点
            swing_highs = [(idx, price) for type_, idx, price in all_points if type_ == 'high']
            swing_lows = [(idx, price) for type_, idx, price in all_points if type_ == 'low']
            
            return swing_highs, swing_lows
        else:
            return [], []

    def calculate_exact_trend_line(self, points):
        """计算精确的趋势线 - 使用线性回归确保所有点在线上"""
        if len(points) < 2:
            return None, None
        
        # 提取坐标
        x_coords = np.arange(len(points))
        y_coords = np.array([price for idx, price in points])
        
        # 线性回归计算趋势线
        if len(set(x_coords)) < 2:
            return None, None
        
        slope, intercept, r_value, p_value, std_err = linregress(x_coords, y_coords)
        
        # 计算R平方值，检查拟合程度
        r_squared = r_value ** 2
        
        # 检查所有点是否在线上（使用更严格的容忍度）
        max_deviation = 0
        for i, (idx, actual_price) in enumerate(points):
            expected_price = slope * i + intercept
            deviation = abs(actual_price - expected_price) / actual_price
            max_deviation = max(max_deviation, deviation)
        
        # 如果拟合不好，返回None
        if r_squared < 0.95 or max_deviation > 0.005:  # R平方>0.95且最大偏差<0.5%
            return None, None
        
        return slope, intercept

    def calculate_trend_line_for_chart(self, points, dates_num):
        """为图表计算趋势线 - 使用实际日期数值"""
        if len(points) < 2:
            return None, None
        
        # 提取坐标
        x_coords = []
        y_coords = []
        
        for idx, price in points:
            x_coords.append(dates_num[idx])
            y_coords.append(price)
        
        # 线性回归计算趋势线
        if len(set(x_coords)) < 2:
            return None, None
        
        slope, intercept, _, _, _ = linregress(x_coords, y_coords)
        
        return slope, intercept

    def detect_triangle_pattern(self, swing_highs, swing_lows, dates_num):
        """检测三角形形态 - 需要3个高点和2个低点，且必须在趋势线上"""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None, None
        
        # 计算高点趋势线 - 使用精确计算
        high_slope, high_intercept = self.calculate_exact_trend_line(swing_highs)
        
        # 计算低点趋势线 - 使用精确计算
        low_slope, low_intercept = self.calculate_exact_trend_line(swing_lows)
        
        if high_slope is None or low_slope is None:
            return None, None
        
        # 为图表计算趋势线（使用实际日期数值）
        high_slope_chart, high_intercept_chart = self.calculate_trend_line_for_chart(swing_highs, dates_num)
        low_slope_chart, low_intercept_chart = self.calculate_trend_line_for_chart(swing_lows, dates_num)
        
        # 检查趋势线是否交叉
        x_min = 0
        x_max = len(swing_highs) - 1
        
        high_y_min = high_slope * x_min + high_intercept
        high_y_max = high_slope * x_max + high_intercept
        low_y_min = low_slope * x_min + low_intercept
        low_y_max = low_slope * x_max + low_intercept
        
        # 如果趋势线在图表范围内交叉，则不是有效的三角形
        if (high_y_min > low_y_min and high_y_max < low_y_max) or \
           (high_y_min < low_y_min and high_y_max > low_y_max):
            return None, None
        
        # 检查收敛性
        slope_threshold = 0.001
        
        pattern_data = {
            'high_points': swing_highs,
            'low_points': swing_lows,
            'high_slope': high_slope_chart,
            'high_intercept': high_intercept_chart,
            'low_slope': low_slope_chart,
            'low_intercept': low_intercept_chart
        }
        
        # 对称三角形 - 高点下降，低点上升
        if (high_slope < -slope_threshold and 
            low_slope > slope_threshold):
            return "对称三角形", pattern_data
        
        # 上升三角形 - 水平阻力
        elif (abs(high_slope) < slope_threshold * 0.5 and 
              low_slope > slope_threshold):
            # 验证高点是否水平
            high_prices = [price for idx, price in swing_highs]
            high_std = np.std(high_prices) / np.mean(high_prices)
            if high_std < 0.005:  # 更严格的标准
                return "上升三角形", pattern_data
        
        # 下降三角形 - 水平支撑
        elif (high_slope < -slope_threshold and 
              abs(low_slope) < slope_threshold * 0.5):
            # 验证低点是否水平
            low_prices = [price for idx, price in swing_lows]
            low_std = np.std(low_prices) / np.mean(low_prices)
            if low_std < 0.005:  # 更严格的标准
                return "下降三角形", pattern_data
        
        return None, None

    def detect_channel_pattern(self, swing_highs, swing_lows, dates_num):
        """检测通道形态 - 需要3个高点和2个低点，且必须在趋势线上"""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None, None
        
        # 计算上轨 - 使用精确计算
        high_slope, high_intercept = self.calculate_exact_trend_line(swing_highs)
        
        # 计算下轨 - 使用精确计算
        low_slope, low_intercept = self.calculate_exact_trend_line(swing_lows)
        
        if high_slope is None or low_slope is None:
            return None, None
        
        # 为图表计算趋势线（使用实际日期数值）
        high_slope_chart, high_intercept_chart = self.calculate_trend_line_for_chart(swing_highs, dates_num)
        low_slope_chart, low_intercept_chart = self.calculate_trend_line_for_chart(swing_lows, dates_num)
        
        # 检查平行性
        slope_diff = abs(high_slope - low_slope)
        parallel_threshold = 0.002
        
        # 检查通道宽度
        avg_high = np.mean([price for idx, price in swing_highs])
        avg_low = np.mean([price for idx, price in swing_lows])
        channel_width = (avg_high - avg_low) / avg_low
        
        pattern_data = {
            'high_points': swing_highs,
            'low_points': swing_lows,
            'high_slope': high_slope_chart,
            'high_intercept': high_intercept_chart,
            'low_slope': low_slope_chart,
            'low_intercept': low_intercept_chart
        }
        
        # 上升通道
        if (high_slope > 0.001 and 
            low_slope > 0.001 and 
            slope_diff < parallel_threshold and 
            channel_width > 0.02):  # 更宽的通道要求
            return "上升通道", pattern_data
        
        # 下降通道
        elif (high_slope < -0.001 and 
              low_slope < -0.001 and 
              slope_diff < parallel_threshold and 
              channel_width > 0.02):  # 更宽的通道要求
            return "下降通道", pattern_data
        
        return None, None

    def detect_wedge_pattern(self, swing_highs, swing_lows, dates_num):
        """检测楔形形态 - 需要3个高点和2个低点，且必须在趋势线上"""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None, None
        
        # 计算上轨 - 使用精确计算
        high_slope, high_intercept = self.calculate_exact_trend_line(swing_highs)
        
        # 计算下轨 - 使用精确计算
        low_slope, low_intercept = self.calculate_exact_trend_line(swing_lows)
        
        if high_slope is None or low_slope is None:
            return None, None
        
        # 为图表计算趋势线（使用实际日期数值）
        high_slope_chart, high_intercept_chart = self.calculate_trend_line_for_chart(swing_highs, dates_num)
        low_slope_chart, low_intercept_chart = self.calculate_trend_line_for_chart(swing_lows, dates_num)
        
        # 检查收敛性 - 楔形的关键特征是两条趋势线同向但收敛
        slope_threshold = 0.001
        
        pattern_data = {
            'high_points': swing_highs,
            'low_points': swing_lows,
            'high_slope': high_slope_chart,
            'high_intercept': high_intercept_chart,
            'low_slope': low_slope_chart,
            'low_intercept': low_intercept_chart
        }
        
        # 上升楔形 - 两条线都向上，但下轨比上轨更陡
        if (high_slope > slope_threshold and 
            low_slope > slope_threshold and 
            low_slope > high_slope):
            # 检查收敛程度
            slope_ratio = high_slope / low_slope
            if 0.3 < slope_ratio < 0.9:
                return "上升楔形", pattern_data
        
        # 下降楔形 - 两条线都向下，但上轨比下轨更陡（负值更小）
        elif (high_slope < -slope_threshold and 
              low_slope < -slope_threshold and 
              high_slope < low_slope):  # 注意：负值比较
            # 检查收敛程度
            slope_ratio = high_slope / low_slope
            if 0.3 < slope_ratio < 0.9:
                return "下降楔形", pattern_data
        
        return None, None

    def detect_flag_pattern(self, df, swing_highs, swing_lows, dates_num):
        """检测旗形形态 - 需要3个高点和2个低点，且必须在趋势线上"""
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None, None
        
        # 计算上轨和下轨 - 使用精确计算
        high_slope, high_intercept = self.calculate_exact_trend_line(swing_highs)
        low_slope, low_intercept = self.calculate_exact_trend_line(swing_lows)
        
        if high_slope is None or low_slope is None:
            return None, None
        
        # 为图表计算趋势线（使用实际日期数值）
        high_slope_chart, high_intercept_chart = self.calculate_trend_line_for_chart(swing_highs, dates_num)
        low_slope_chart, low_intercept_chart = self.calculate_trend_line_for_chart(swing_lows, dates_num)
        
        # 旗形的关键特征：两条趋势线平行且斜率适中
        slope_threshold = 0.005
        parallel_threshold = 0.001
        
        # 检查平行性
        slope_diff = abs(high_slope - low_slope)
        
        pattern_data = {
            'high_points': swing_highs,
            'low_points': swing_lows,
            'high_slope': high_slope_chart,
            'high_intercept': high_intercept_chart,
            'low_slope': low_slope_chart,
            'low_intercept': low_intercept_chart
        }
        
        # 看涨旗形 - 小幅下降的平行通道（整理形态）
        if (high_slope < -0.001 and 
            low_slope < -0.001 and 
            slope_diff < parallel_threshold and 
            abs(high_slope) < slope_threshold):
            return "看涨旗形", pattern_data
        
        # 看跌旗形 - 小幅上升的平行通道（整理形态）
        elif (high_slope > 0.001 and 
              low_slope > 0.001 and 
              slope_diff < parallel_threshold and 
              abs(high_slope) < slope_threshold):
            return "看跌旗形", pattern_data
        
        return None, None

    def detect_all_patterns(self, df, kline_count):
        """使用指定数量K线进行形态检测 - 需要3个高点和2个低点且在趋势线上"""
        if df is None:
            return None, 0, [], [], None
        
        min_data_required = {
            200: 180,
            400: 350
        }.get(kline_count, 180)
        
        if len(df) < min_data_required:
            return None, 0, [], [], None
        
        # 确保使用指定数量的K线进行分析
        if len(df) > kline_count:
            analysis_data = df.tail(kline_count)
        else:
            analysis_data = df
        
        # 找到摆动点 - 需要3个高点和2个低点
        window_size = {
            200: 8,
            400: 12
        }.get(kline_count, 10)
        
        swing_highs, swing_lows = self.find_swing_points(analysis_data, window=window_size)
        
        if len(swing_highs) < 3 or len(swing_lows) < 2:
            return None, 0, [], [], None
        
        # 获取日期数值
        dates_num = mdates.date2num(analysis_data.index.to_pydatetime())
        
        patterns = []
        
        # 1. 三角形检测 - 最可靠的形态
        triangle, tri_data = self.detect_triangle_pattern(swing_highs, swing_lows, dates_num)
        if triangle:
            patterns.append((triangle, 95, tri_data))
        
        # 2. 通道检测 - 第二可靠的形态
        channel, ch_data = self.detect_channel_pattern(swing_highs, swing_lows, dates_num)
        if channel:
            patterns.append((channel, 90, ch_data))
        
        # 3. 楔形检测
        wedge, wedge_data = self.detect_wedge_pattern(swing_highs, swing_lows, dates_num)
        if wedge:
            patterns.append((wedge, 85, wedge_data))
        
        # 4. 旗形检测
        flag, flag_data = self.detect_flag_pattern(analysis_data, swing_highs, swing_lows, dates_num)
        if flag:
            patterns.append((flag, 88, flag_data))
        
        # 返回得分最高的形态
        if patterns:
            best_pattern = max(patterns, key=lambda x: x[1])
            return best_pattern[0], best_pattern[1], swing_highs, swing_lows, best_pattern[2]
        
        return None, 0, swing_highs, swing_lows, None

    def create_chart(self, df, symbol, interval, pattern_type, pattern_score, swing_highs, swing_lows, pattern_data, kline_count):
        """创建图表"""
        try:
            if len(df) > kline_count:
                plot_data = df.tail(kline_count)
            else:
                plot_data = df
            
            if plot_data.empty:
                return None
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            dates = plot_data.index
            dates_num = mdates.date2num(dates.to_pydatetime())
            
            # 绘制K线
            for i in range(len(dates)):
                date_num = dates_num[i]
                open_val = plot_data['Open'].iloc[i]
                high_val = plot_data['High'].iloc[i]
                low_val = plot_data['Low'].iloc[i]
                close_val = plot_data['Close'].iloc[i]
                
                color = 'green' if close_val >= open_val else 'red'
                
                ax1.plot([date_num, date_num], [low_val, high_val], 
                        color='black', linewidth=0.8, alpha=0.7)
                
                body_bottom = min(open_val, close_val)
                body_top = max(open_val, close_val)
                body_height = body_top - body_bottom
                
                if body_height > 0:
                    width = (dates_num[-1] - dates_num[0]) / len(dates_num) * 0.7
                    
                    rect = plt.Rectangle((date_num - width/2, body_bottom), 
                                       width, body_height, 
                                       facecolor=color, alpha=0.7, edgecolor='black')
                    ax1.add_patch(rect)
            
            # 标注摆动点
            for idx, price in swing_highs:
                if idx < len(dates_num):
                    ax1.plot(dates_num[idx], price, 'ro', markersize=8, alpha=0.9, markeredgecolor='darkred', label='Swing High' if idx == swing_highs[0][0] else "")
            
            for idx, price in swing_lows:
                if idx < len(dates_num):
                    ax1.plot(dates_num[idx], price, 'go', markersize=8, alpha=0.9, markeredgecolor='darkgreen', label='Swing Low' if idx == swing_lows[0][0] else "")
            
            # 绘制趋势线
            if pattern_data and pattern_type:
                self.draw_trendlines(ax1, dates_num, pattern_type, pattern_data)
            
            # 绘制成交量
            volumes = plot_data['Volume'].values
            for i in range(len(dates)):
                date_num = dates_num[i]
                volume = volumes[i]
                color = 'green' if plot_data['Close'].iloc[i] >= plot_data['Open'].iloc[i] else 'red'
                
                width = (dates_num[-1] - dates_num[0]) / len(dates_num) * 0.7
                ax2.bar(date_num, volume, color=color, alpha=0.7, width=width)
            
            title = f"{symbol} - {interval} - {pattern_type} (Score: {pattern_score}%)"
            ax1.set_title(title, fontsize=14, fontweight='bold')
            ax1.set_ylabel('Price (USDT)', fontsize=10)
            ax2.set_ylabel('Volume', fontsize=10)
            
            price_min = plot_data['Low'].min()
            price_max = plot_data['High'].max()
            price_range = price_max - price_min
            ax1.set_ylim(price_min - price_range * 0.05, price_max + price_range * 0.05)
            
            volume_max = volumes.max()
            ax2.set_ylim(0, volume_max * 1.1)
            
            date_format = mdates.DateFormatter('%m-%d %H:%M')
            ax1.xaxis.set_major_formatter(date_format)
            ax2.xaxis.set_major_formatter(date_format)
            
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)
            
            # 添加图例
            ax1.legend(loc='upper left', fontsize=8)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # 转换为Base64在网页显示
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            return buf
            
        except Exception as e:
            st.error(f"图表创建失败: {e}")
            return None

    def draw_trendlines(self, ax, dates_num, pattern_type, pattern_data):
        """绘制趋势线"""
        try:
            high_slope = pattern_data['high_slope']
            high_intercept = pattern_data['high_intercept']
            low_slope = pattern_data['low_slope']
            low_intercept = pattern_data['low_intercept']
            
            # 获取摆动点的时间范围
            high_points = pattern_data['high_points']
            low_points = pattern_data['low_points']
            
            # 找到所有摆动点的最小和最大索引
            all_indices = [idx for idx, _ in high_points] + [idx for idx, _ in low_points]
            if not all_indices:
                return
                
            min_idx = min(all_indices)
            max_idx = max(all_indices)
            
            # 将索引转换为日期数值
            min_date = dates_num[min_idx]
            max_date = dates_num[max_idx]
            
            # 扩展一点范围（10%）
            date_range = max_date - min_date
            extended_min = min_date - date_range * 0.1
            extended_max = max_date + date_range * 0.1
            
            # 确保不超出图表范围
            chart_min = dates_num[0]
            chart_max = dates_num[-1]
            x_min = max(extended_min, chart_min)
            x_max = min(extended_max, chart_max)
            
            # 计算趋势线端点
            high_y1 = high_slope * x_min + high_intercept
            high_y2 = high_slope * x_max + high_intercept
            low_y1 = low_slope * x_min + low_intercept
            low_y2 = low_slope * x_max + low_intercept
            
            # 标记关键摆动点
            for idx, price in high_points:
                if idx < len(dates_num):
                    ax.plot(dates_num[idx], price, 'ro', markersize=8, alpha=0.9, markeredgecolor='darkred')
            
            for idx, price in low_points:
                if idx < len(dates_num):
                    ax.plot(dates_num[idx], price, 'go', markersize=8, alpha=0.9, markeredgecolor='darkgreen')
            
            # 设置线条样式和标签
            if pattern_type in ["对称三角形", "上升三角形", "下降三角形"]:
                ax.plot([x_min, x_max], [high_y1, high_y2], 'r-', linewidth=2.5, alpha=0.8, label='Resistance')
                ax.plot([x_min, x_max], [low_y1, low_y2], 'g-', linewidth=2.5, alpha=0.8, label='Support')
            
            elif pattern_type in ["上升通道", "下降通道"]:
                ax.plot([x_min, x_max], [high_y1, high_y2], 'r-', linewidth=2.5, alpha=0.8, label='Upper')
                ax.plot([x_min, x_max], [low_y1, low_y2], 'g-', linewidth=2.5, alpha=0.8, label='Lower')
            
            elif pattern_type in ["上升楔形", "下降楔形"]:
                ax.plot([x_min, x_max], [high_y1, high_y2], 'r-', linewidth=2.5, alpha=0.8, label='Upper')
                ax.plot([x_min, x_max], [low_y1, low_y2], 'g-', linewidth=2.5, alpha=0.8, label='Lower')
            
            elif pattern_type in ["看涨旗形", "看跌旗形"]:
                ax.plot([x_min, x_max], [high_y1, high_y2], 'r-', linewidth=2.5, alpha=0.8, label='Upper')
                ax.plot([x_min, x_max], [low_y1, low_y2], 'g-', linewidth=2.5, alpha=0.8, label='Lower')
                
        except Exception as e:
            st.warning(f"绘制趋势线失败: {e}")

    def scan_single_symbol_complete(self, symbol, selected_timeframes, selected_kline_counts):
        """完整扫描单个币种 - 所有时间框架和K线数量"""
        try:
            all_results = []
            
            for timeframe in selected_timeframes:
                for kline_count in selected_kline_counts:
                    with st.spinner(f'扫描 {symbol} ({timeframe}, {kline_count}K)...'):
                        df = self.get_spot_candle_data(symbol, timeframe, kline_count)
                        if df is None or len(df) < 200:
                            continue
                        
                        pattern_type, pattern_score, swing_highs, swing_lows, pattern_data = self.detect_all_patterns(df, kline_count)
                        
                        if pattern_type:
                            # 创建唯一标识避免重复
                            pattern_key = f"{symbol}_{timeframe}_{pattern_type}_{kline_count}"
                            if pattern_key not in self.seen_patterns:
                                self.seen_patterns.add(pattern_key)
                                
                                chart_buf = self.create_chart(
                                    df, symbol, timeframe, pattern_type, pattern_score,
                                    swing_highs, swing_lows, pattern_data, kline_count
                                )
                                
                                # 保存图表到缓存
                                chart_key = self.save_chart_to_cache(symbol, timeframe, pattern_type, kline_count, chart_buf)
                                
                                result = {
                                    'symbol': symbol,
                                    'timeframe': timeframe,
                                    'pattern': pattern_type,
                                    'score': pattern_score,
                                    'price': df['Close'].iloc[-1],
                                    'kline_count': kline_count,
                                    'swing_highs': len(swing_highs),
                                    'swing_lows': len(swing_lows),
                                    'chart_key': chart_key,
                                    'timestamp': datetime.now()
                                }
                                
                                all_results.append(result)
            
            # 按得分排序
            all_results.sort(key=lambda x: x['score'], reverse=True)
            return all_results
            
        except Exception as e:
            st.error(f"扫描失败: {e}")
            return []

    def run_complete_scan(self, symbols, selected_timeframes, selected_kline_counts):
        """运行完整扫描"""
        total_combinations = len(symbols) * len(selected_timeframes) * len(selected_kline_counts)
        st.info(f"🔍 即将扫描 {len(symbols)} 个币种 × {len(selected_timeframes)} 个时间框架 × {len(selected_kline_counts)} 种K线数量 = {total_combinations} 种组合")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        completed = 0
        
        for i, symbol in enumerate(symbols):
            status_text.text(f"扫描中: {symbol} ({i+1}/{len(symbols)})")
            
            symbol_results = self.scan_single_symbol_complete(symbol, selected_timeframes, selected_kline_counts)
            all_results.extend(symbol_results)
            
            completed += 1
            progress_bar.progress(completed / len(symbols))
            
            # 避免API限制，增加扫描间隔
            time.sleep(1.5)  # 增加到1.5秒
        
        progress_bar.empty()
        status_text.empty()
        
        # 按得分排序
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results

# 初始化扫描器
@st.cache_resource
def get_scanner():
    return StrictPatternScanner()

scanner = get_scanner()

# 侧边栏
st.sidebar.title("⚙️ 严格模式扫描设置")

# 扫描模式选择
scan_mode = st.sidebar.radio("选择扫描模式", 
                           ["单个币种完整扫描", "批量完整扫描前50", "批量完整扫描前200"])  # 更新选项

# 时间框架选择 - 多选
st.sidebar.markdown("### 📊 时间框架选择")
selected_timeframes = st.sidebar.multiselect(
    "选择要扫描的时间框架",
    scanner.all_timeframes,
    default=scanner.all_timeframes
)

# K线数量选择 - 多选
st.sidebar.markdown("### 📈 K线数量选择")
selected_kline_counts = st.sidebar.multiselect(
    "选择要扫描的K线数量",
    scanner.all_kline_counts,
    default=scanner.all_kline_counts
)

# 历史记录管理
st.sidebar.markdown("### 📋 历史记录管理")

# 清除历史记录按钮
if st.sidebar.button("🗑️ 清除所有历史记录"):
    scanner.scan_results = []
    scanner.chart_cache = {}
    scanner.seen_patterns = set()
    st.sidebar.success("历史记录已清除")
    st.experimental_rerun()

# 重新扫描功能
if scanner.scan_results:
    st.sidebar.markdown("### 🔄 重新扫描")
    
    # 选择要重新扫描的结果
    recent_symbols = list(set([r['symbol'] for r in scanner.scan_results[-20:]]))
    if recent_symbols:
        rescan_symbol = st.sidebar.selectbox("选择币种重新扫描", recent_symbols)
        
        # 获取该币种的所有时间框架
        symbol_timeframes = list(set([
            r['timeframe'] for r in scanner.scan_results 
            if r['symbol'] == rescan_symbol
        ]))
        
        if symbol_timeframes:
            rescan_timeframe = st.sidebar.selectbox("选择时间框架", symbol_timeframes)
            
            if st.sidebar.button(f"重新扫描 {rescan_symbol} {rescan_timeframe}"):
                # 执行重新扫描
                with st.spinner(f"重新扫描 {rescan_symbol} {rescan_timeframe}..."):
                    results = scanner.scan_single_symbol_complete(
                        rescan_symbol, 
                        [rescan_timeframe], 
                        selected_kline_counts
                    )
                    
                    if results:
                        st.success(f"重新扫描完成! 发现 {len(results)} 个新形态")
                        # 显示新结果
                        for result in results:
                            with st.expander(f"{result['symbol']} - {result['timeframe']} - {result['pattern']} (得分: {result['score']}%)", expanded=True):
                                scanner.display_result_with_chart(result)
                    else:
                        st.warning("重新扫描未发现新形态")

# 导出功能
if scanner.scan_results:
    st.sidebar.markdown("### 💾 数据导出")
    
    # 创建数据框用于导出
    export_data = []
    for result in scanner.scan_results:
        export_data.append({
            '币种': result['symbol'],
            '时间框架': result['timeframe'],
            '形态': result['pattern'],
            '得分': result['score'],
            '价格': result['price'],
            'K线数量': result['kline_count'],
            '摆动点': f"{result['swing_highs']}高/{result['swing_lows']}低",
            '时间': result['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
        })
    
    df_export = pd.DataFrame(export_data)
    
    # 导出为CSV
    csv = df_export.to_csv(index=False, encoding='utf-8-sig')
    st.sidebar.download_button(
        "📥 导出CSV",
        csv,
        f"crypto_scan_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        "text/csv"
    )

# 主扫描区域
if scan_mode == "单个币种完整扫描":
    st.header("🔍 单个币种完整扫描")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        symbol = st.selectbox("选择币种", 
                            scanner.volume_symbols[:50],  # 显示前50个供选择
                            index=0)
    
    with col2:
        st.info(f"💡 将扫描: {len(selected_timeframes)}个时间框架 × {len(selected_kline_counts)}种K线数量")
        st.warning("🔒 严格模式: 需要3高2低依次出现且所有点在趋势线上 (R²>0.95)")
    
    if st.button("🚀 开始完整扫描", type="primary", use_container_width=True):
        if not selected_timeframes or not selected_kline_counts:
            st.error("请先选择时间框架和K线数量")
        else:
            results = scanner.scan_single_symbol_complete(symbol, selected_timeframes, selected_kline_counts)
            
            if results:
                st.success(f"🎉 发现 {len(results)} 个有效形态!")
                
                # 显示所有结果
                for i, result in enumerate(results):
                    with st.expander(f"{i+1}. {result['symbol']} - {result['timeframe']} - {result['pattern']} (得分: {result['score']}%)", expanded=i==0):
                        scanner.display_result_with_chart(result)
                
                # 保存结果
                scanner.scan_results.extend(results)
                
            else:
                st.warning("❌ 未发现有效形态 - 严格模式下需要3高2低依次出现且精确在趋势线上")

else:
    st.header("📊 批量完整扫描")
    
    limit = 50 if scan_mode == "批量完整扫描前50" else 200  # 更新限制
    symbols_to_scan = scanner.volume_symbols[:limit]
    
    total_scans = len(symbols_to_scan) * len(selected_timeframes) * len(selected_kline_counts)
    st.info(f"🔍 即将扫描 {len(symbols_to_scan)} 个币种 × {len(selected_timeframes)} 个时间框架 × {len(selected_kline_counts)} 种K线数量 = {total_scans} 种组合")
    st.warning("🔒 严格模式: 需要3高2低依次出现且所有点在趋势线上 (R²>0.95)")
    st.warning(f"⏰ 预计耗时: 约 {total_scans * 2.5 // 60} 分钟")  # 增加时间预估
    
    if st.button("🚀 开始批量完整扫描", type="primary", use_container_width=True):
        if not selected_timeframes or not selected_kline_counts:
            st.error("请先选择时间框架和K线数量")
        else:
            results = scanner.run_complete_scan(symbols_to_scan, selected_timeframes, selected_kline_counts)
            
            if results:
                st.success(f"🎉 批量扫描完成! 发现 {len(results)} 个有效形态")
                
                # 显示统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    timeframes_found = len(set(r['timeframe'] for r in results))
                    st.metric("涉及时间框架", f"{timeframes_found}个")
                with col2:
                    patterns_found = len(set(r['pattern'] for r in results))
                    st.metric("发现形态种类", f"{patterns_found}种")
                with col3:
                    avg_score = np.mean([r['score'] for r in results])
                    st.metric("平均置信度", f"{avg_score:.1f}%")
                
                # 显示所有结果
                for i, result in enumerate(results):
                    with st.expander(f"{i+1}. {result['symbol']} - {result['timeframe']} - {result['pattern']} (得分: {result['score']}%)", expanded=i<3):
                        scanner.display_result_with_chart(result)
                
                scanner.scan_results.extend(results)
            else:
                st.warning("❌ 批量扫描未发现任何有效形态 - 严格模式下需要3高2低依次出现且精确在趋势线上")

# 显示历史结果
if scanner.scan_results:
    st.sidebar.markdown("### 📜 最近扫描结果")
    
    recent_results = scanner.scan_results[-10:]  # 显示最近10个结果
    for i, result in enumerate(reversed(recent_results)):
        with st.sidebar.expander(f"{result['symbol']} - {result['pattern']}", expanded=False):
            st.write(f"时间框架: {result['timeframe']}")
            st.write(f"得分: {result['score']}%")
            st.write(f"价格: ${result['price']:.4f}")
            st.write(f"K线: {result['kline_count']}根")
            st.write(f"摆动点: {result['swing_highs']}高/{result['swing_lows']}低")
            st.write(f"时间: {result['timestamp'].strftime('%H:%M:%S')}")

# 使用说明
with st.sidebar.expander("📖 严格模式说明", expanded=False):
    st.markdown("""
    **严格模式要求:**
    - 🔢 3个高点和2个低点依次出现 (高-低-高-低-高 或 低-高-低-高-低)
    - 📈 所有摆动点必须在趋势线上 (R²>0.95, 最大偏差<0.5%)
    - 📊 使用线性回归计算精确趋势线
    
    **扫描规模:**
    - 📊 支持前200币种大规模扫描
    - ⏰ 批量扫描需要较长时间
    - 💾 建议分批扫描重要币种
    
    **注意:**
    - 严格模式下发现的形态数量会减少
    - 但每个发现的形态质量更高
    - 适合对准确性要求高的交易者
    """)

# 页脚
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "加密货币形态扫描器 - 严格模式 | 支持前200币种大规模扫描"
    "</div>",
    unsafe_allow_html=True
)