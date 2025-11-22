import requests
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 支援中文顯示
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常顯示負號

# 設定參數
fund_id = "24" # 復華基金
s_date = "2023/03/08"  # 改用 YYYY-MM-DD 格式
e_date = "2025/11/22"
s_date_yf = "2023-03-08"
e_date_yf = "2025-11-22"
url = f"https://www.fhtrust.com.tw/api/fundNav?fundID={fund_id}&sDate={s_date}&eDate={e_date}"

# 1. 取得基金資料 (Get Fund Data)
try:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
    }
    response = requests.get(url, headers=headers)

    if not response.text or response.text.strip() == "":
        print("錯誤：API 回傳空白內容")
        exit()

    # 設定正確的編碼並解析 JSON
    response.encoding = 'utf-8'
    try:
        data = response.json()
    except Exception as json_error:
        print(f"JSON 解析錯誤: {json_error}")
        # 嘗試修復可能的 JSON 格式問題
        import json
        try:
            # 有時候回傳的 JSON 可能有多餘的字符，嘗試清理
            text = response.text.strip()
            data = json.loads(text)
        except:
            print("無法解析 JSON，請檢查 API 回傳格式")
            exit()

    # API 回傳的結構是 {"result": [...]}
    if isinstance(data, dict) and 'result' in data:
        df_fund = pd.DataFrame(data['result'])
    else:
        df_fund = pd.DataFrame(data)
    
    # 清洗資料：確保有日期與淨值欄位
    # 復華 API 通常欄位名稱可能為 'Date'/'NAVDATE' 和 'Nav'/'NAV'
    # 這裡做欄位名稱標準化處理
    df_fund.columns = [c.lower() for c in df_fund.columns]
    date_col = [c for c in df_fund.columns if 'date' in c][0]
    nav_col = [c for c in df_fund.columns if 'nav' in c][0]
    
    df_fund[date_col] = pd.to_datetime(df_fund[date_col])
    df_fund[nav_col] = pd.to_numeric(df_fund[nav_col])
    df_fund = df_fund.set_index(date_col).sort_index()
    df_fund = df_fund.rename(columns={nav_col: 'Fund_NAV'})
    
    # 計算基金日報酬率
    df_fund['Fund_Ret'] = df_fund['Fund_NAV'].pct_change()
    
    print(f"成功取得基金資料，期間: {df_fund.index.min().date()} 至 {df_fund.index.max().date()}")

except Exception as e:
    print(f"取得基金資料失敗: {e}")
    exit()

# 2. 取得市場基準資料 (Get Benchmark Data) - 用於計算 Beta
# 假設基準為台灣加權指數 (^TWII)
benchmark_symbol = "^TWII" 
print(f"正在下載市場基準 ({benchmark_symbol}) 資料...")

df_bench = yf.download(benchmark_symbol, start=s_date_yf, end=e_date_yf, progress=False, auto_adjust=True)
if not df_bench.empty:
    # yfinance 下載的資料，'Close' 可能是 MultiIndex 或 Series，需統一處理
    if isinstance(df_bench.columns, pd.MultiIndex):
        # 如果是 MultiIndex (例如包含 Ticker)，取 'Close' 並選取第一個欄位
        try:
            bench_close = df_bench['Close'].iloc[:, 0]
        except:
             bench_close = df_bench['Close']
    else:
        bench_close = df_bench['Close']

    # 計算市場日報酬率
    df_bench_ret = bench_close.pct_change().rename("Market_Ret")
    
    # 合併基金與市場資料 (取交集日期)
    df_merge = pd.concat([df_fund['Fund_Ret'], df_bench_ret], axis=1).dropna()
else:
    print("無法取得市場資料，Beta 將無法計算。")
    df_merge = df_fund.dropna()

# 3. 計算各項指標 (Calculate Metrics)

# 設定無風險利率 (Risk-Free Rate)，假設年化 1.5%
rf_annual = 0.015
rf_daily = rf_annual / 252

# A. Sharpe Ratio (夏普值)
# 公式: (年化報酬率 - 無風險利率) / 年化標準差
mean_ret = df_merge['Fund_Ret'].mean() * 252
std_dev = df_merge['Fund_Ret'].std() * (252 ** 0.5)
sharpe_ratio = (mean_ret - rf_annual) / std_dev

# B. Beta (貝他值)
# 公式: Cov(基金, 市場) / Var(市場)
if 'Market_Ret' in df_merge.columns:
    covariance_matrix = np.cov(df_merge['Fund_Ret'], df_merge['Market_Ret'])
    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1]
else:
    beta = np.nan

# C. Max Drawdown (最大回撤)
# 公式: (NAV - 歷史最高NAV) / 歷史最高NAV 的最小值
rolling_max = df_fund['Fund_NAV'].cummax()
drawdown = (df_fund['Fund_NAV'] - rolling_max) / rolling_max
max_drawdown = drawdown.min()

# D. Sortino Ratio (索提諾指標)
# 公式: (年化報酬率 - 無風險利率) / 下行標準差
# 下行標準差: 只取負報酬 (或低於無風險利率的報酬) 計算標準差
downside_returns = df_merge.loc[df_merge['Fund_Ret'] < 0, 'Fund_Ret']
downside_std = downside_returns.std() * (252 ** 0.5)
sortino_ratio = (mean_ret - rf_annual) / downside_std

# 4. 輸出結果
print("-" * 30)
print(f"基金 ID: {fund_id} (復華華人世紀基金 預估)")
print(f"統計期間: {s_date} ~ {e_date}")
print("-" * 30)
print(f"1. Sharpe Ratio (夏普值): {sharpe_ratio:.4f}")
print(f"2. Beta (貝他值)        : {beta:.4f}")
print(f"3. Max Drawdown (最大回撤): {max_drawdown:.2%}")
print(f"4. Sortino Ratio (索提諾): {sortino_ratio:.4f}")
print("-" * 30)

# 5. 蒙地卡羅模擬 (Monte Carlo Simulation)
print("\n正在執行蒙地卡羅模擬...")

# 蒙地卡羅參數設定
n_simulations = 10000  # 模擬次數
n_days = 252  # 模擬天數 (1年交易日)
initial_nav = df_fund['Fund_NAV'].iloc[0]  # 使用最舊淨值作為起始點

# 基於歷史數據計算參數
daily_return = df_merge['Fund_Ret'].mean()  # 日均報酬率
daily_volatility = df_merge['Fund_Ret'].std()  # 日波動率

# 蒙地卡羅模擬函數
def monte_carlo_simulation(initial_price, daily_return, daily_volatility, n_days, n_simulations):
    """
    使用幾何布朗運動 (Geometric Brownian Motion) 進行蒙地卡羅模擬

    參數:
        initial_price: 起始淨值
        daily_return: 日均報酬率
        daily_volatility: 日波動率
        n_days: 模擬天數
        n_simulations: 模擬次數

    回傳:
        simulation_results: 形狀為 (n_simulations, n_days+1) 的陣列
    """
    simulation_results = np.zeros((n_simulations, n_days + 1))
    simulation_results[:, 0] = initial_price

    for sim in range(n_simulations):
        for day in range(1, n_days + 1):
            # 產生隨機變動 (標準常態分佈)
            random_shock = np.random.normal(0, 1)
            # 幾何布朗運動公式: S(t+1) = S(t) * exp((μ - σ²/2)Δt + σ√Δt * Z)
            drift = (daily_return - 0.5 * daily_volatility**2)
            diffusion = daily_volatility * random_shock
            simulation_results[sim, day] = simulation_results[sim, day - 1] * np.exp(drift + diffusion)

    return simulation_results

# 執行模擬
simulation_results = monte_carlo_simulation(initial_nav, daily_return, daily_volatility, n_days, n_simulations)

# 計算統計指標
final_prices = simulation_results[:, -1]  # 所有模擬的最終淨值
mean_final_price = np.mean(final_prices)
median_final_price = np.median(final_prices)
std_final_price = np.std(final_prices)

# 計算置信區間
confidence_95 = np.percentile(final_prices, [2.5, 97.5])
confidence_99 = np.percentile(final_prices, [0.5, 99.5])

# 計算 VaR (Value at Risk) - 在給定信心水準下的最大損失
var_95 = np.percentile(final_prices, 5)  # 95% VaR
var_99 = np.percentile(final_prices, 1)  # 99% VaR

# 計算 CVaR (Conditional VaR / Expected Shortfall) - 超過VaR的平均損失
cvar_95 = final_prices[final_prices <= var_95].mean()
cvar_99 = final_prices[final_prices <= var_99].mean()

# 計算預期報酬率
expected_return = (mean_final_price - initial_nav) / initial_nav

print("\n" + "=" * 50)
print("蒙地卡羅模擬結果")
print("=" * 50)
print(f"模擬次數: {n_simulations:,}")
print(f"模擬期間: {n_days} 交易日 (約 {n_days/252:.1f} 年)")
print(f"起始淨值: {initial_nav:.4f}")
print("-" * 50)
print(f"預期最終淨值 (平均): {mean_final_price:.4f}")
print(f"預期最終淨值 (中位數): {median_final_price:.4f}")
print(f"標準差: {std_final_price:.4f}")
print(f"預期報酬率: {expected_return:.2%}")
print("-" * 50)
print(f"95% 置信區間: [{confidence_95[0]:.4f}, {confidence_95[1]:.4f}]")
print(f"99% 置信區間: [{confidence_99[0]:.4f}, {confidence_99[1]:.4f}]")
print("-" * 50)
print(f"VaR (95%): {var_95:.4f} (損失: {(var_95-initial_nav)/initial_nav:.2%})")
print(f"VaR (99%): {var_99:.4f} (損失: {(var_99-initial_nav)/initial_nav:.2%})")
print(f"CVaR (95%): {cvar_95:.4f} (平均損失: {(cvar_95-initial_nav)/initial_nav:.2%})")
print(f"CVaR (99%): {cvar_99:.4f} (平均損失: {(cvar_99-initial_nav)/initial_nav:.2%})")
print("=" * 50)

# 6. 視覺化模擬結果
print("\n正在生成圖表...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('蒙地卡羅模擬結果分析', fontsize=16, fontweight='bold')

# 圖1: 模擬路徑 (顯示100條隨機路徑)
ax1 = axes[0, 0]
sample_simulations = np.random.choice(n_simulations, size=100, replace=False)
for sim_idx in sample_simulations:
    ax1.plot(simulation_results[sim_idx, :], alpha=0.3, linewidth=0.5, color='steelblue')
# 繪製平均路徑
mean_path = simulation_results.mean(axis=0)
ax1.plot(mean_path, color='red', linewidth=2, label='平均路徑')
ax1.axhline(y=initial_nav, color='green', linestyle='--', linewidth=1.5, label='起始淨值')
ax1.set_xlabel('交易日')
ax1.set_ylabel('基金淨值')
ax1.set_title('模擬路徑 (100條樣本路徑)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 圖2: 最終淨值分布直方圖
ax2 = axes[0, 1]
ax2.hist(final_prices, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
ax2.axvline(mean_final_price, color='red', linestyle='--', linewidth=2, label=f'平均值: {mean_final_price:.2f}')
ax2.axvline(median_final_price, color='orange', linestyle='--', linewidth=2, label=f'中位數: {median_final_price:.2f}')
ax2.axvline(initial_nav, color='green', linestyle='--', linewidth=2, label=f'起始值: {initial_nav:.2f}')
ax2.set_xlabel('最終淨值')
ax2.set_ylabel('頻率')
ax2.set_title('最終淨值分布直方圖')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 圖3: 置信區間與 VaR 可視化
ax3 = axes[1, 0]
percentiles = np.arange(0, 101, 1)
percentile_values = np.percentile(final_prices, percentiles)
ax3.plot(percentiles, percentile_values, linewidth=2, color='navy')
ax3.fill_between(percentiles, percentile_values.min(), percentile_values,
                 where=(percentiles <= 5), alpha=0.3, color='red', label='VaR 95%')
ax3.fill_between(percentiles, percentile_values.min(), percentile_values,
                 where=(percentiles <= 1), alpha=0.5, color='darkred', label='VaR 99%')
ax3.axhline(y=initial_nav, color='green', linestyle='--', linewidth=1.5, label='起始淨值')
ax3.axhline(y=var_95, color='red', linestyle=':', linewidth=1.5)
ax3.axhline(y=var_99, color='darkred', linestyle=':', linewidth=1.5)
ax3.set_xlabel('百分位數 (%)')
ax3.set_ylabel('淨值')
ax3.set_title('百分位數分布與 VaR')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 圖4: 報酬率分布
ax4 = axes[1, 1]
returns = (final_prices - initial_nav) / initial_nav * 100  # 轉換為百分比
ax4.hist(returns, bins=100, alpha=0.7, color='lightgreen', edgecolor='black')
ax4.axvline(returns.mean(), color='red', linestyle='--', linewidth=2,
           label=f'平均報酬率: {returns.mean():.2f}%')
ax4.axvline(0, color='black', linestyle='-', linewidth=1.5, label='損益兩平')
# 標示風險區域
ax4.axvline(np.percentile(returns, 5), color='orange', linestyle=':', linewidth=2,
           label=f'VaR 95%: {np.percentile(returns, 5):.2f}%')
ax4.axvline(np.percentile(returns, 1), color='red', linestyle=':', linewidth=2,
           label=f'VaR 99%: {np.percentile(returns, 1):.2f}%')
ax4.set_xlabel('報酬率 (%)')
ax4.set_ylabel('頻率')
ax4.set_title('預期報酬率分布')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_simulation.png', dpi=300, bbox_inches='tight')
print("圖表已儲存為 'monte_carlo_simulation.png'")
plt.show()

print("\n蒙地卡羅模擬完成！")