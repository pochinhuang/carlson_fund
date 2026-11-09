# Carlson Fund 本機完整測試包

這個資料夾包含更新後的三份原始資料、PostgreSQL 建表與匯入、data_access、原有 Streamlit 首頁與文件頁，以及每日行情更新程式。

## 1. 第一次啟動

電腦只需要 **Docker Desktop（開啟中）＋ Python 3.9 以上**。應用程式的 Python 套件會在容器內安裝，不必安裝到你的 base 環境。

建議把本資料夾解壓縮到原本的 `carlson_funds` 專案內，讓啟動程式能找到原有 `imgs`。在含有 `start.py` 的資料夾開啟 PowerShell：

```powershell
python start.py
```

這個指令會：

1. 從 `.env.example` 建立 `.env`，自動產生資料庫密碼。
2. 選擇可用的本機 port；資料庫預設從 `25432` 開始，網站從 `8501` 開始。
3. 從附近的原專案複製 `imgs`，保留原有圖片。
4. 建立 Docker image、啟動 PostgreSQL、建表並匯入三份 CSV。
5. 檢查持倉、全部因子模型、Brinson 加總與風險貢獻加總。
6. 啟動 Streamlit 與獨立的每日更新服務，印出網站網址。

第一次需要下載映像與套件。重跑會比對三份資料的 SHA-256；資料未變時跳過匯入，不會清除每日已新增的價格。

### 原始 logo

你提供的來源沒有 `imgs/carlson_funds_enterprise_logo.jpeg`。為維持原外觀，本包沒有自行換圖或改寫 `home.py`。啟動程式會檢查上層與上兩層的 `imgs`。若沒有找到，可指定原專案：

```powershell
python start.py --assets-from "C:\Users\user\Desktop\UMN\FM2026 Spring\retry\carlson_funds"
```

也可以直接把你原本的 `imgs` 資料夾複製進來。若缺少 logo，資料庫與檢查仍會完成，啟動程式會請你提供圖片後再啟動首頁，避免開啟一個缺圖的頁面。

若只想先測資料庫：

```powershell
python start.py --database-only
```

本包使用獨立的 Compose project／volume：`carlson_fund_local`。不會停用或刪除你之前建立的資料庫。

## 2. Alpaca key 填在哪裡

第一次啟動後，開啟**本資料夾的 `.env`**，填入：

```dotenv
ALPACA_API_KEY=你的APIKey
ALPACA_SECRET_KEY=你的SecretKey
ALPACA_FEED=sip
```

也可以先將 `.env.example` 複製為 `.env` 後填寫；空白的 `POSTGRES_PASSWORD` 仍會由 `start.py` 自動產生。

不用改 Python 程式，不要把 key 放進 CSV、文件頁或上傳到 Git。程式只呼叫市場資料 API，不包含下單功能。

確認設定已讀到：

```powershell
docker compose run --rm worker python update_daily_prices.py --check-config
```

只會顯示 key 是否存在，不會印出內容。手動更新一次：

```powershell
docker compose run --rm worker python update_daily_prices.py
```

預設處理設定時區中的「昨天」。也可以重試指定的過去日期：

```powershell
docker compose run --rm worker python update_daily_prices.py --date 2026-11-05
```

填寫或更換 key 後，讓排程容器重新讀取設定：

```powershell
docker compose up -d --force-recreate scheduler
```

`sip` 是預設，使用明確截止於過去日期的 historical bars；若你的帳戶／資料權限回覆 403，請先核對 key 與 feed 權限。可改 `ALPACA_FEED=iex` 測試，但 IEX 是單一交易所，其 Close 不一定等於全市場的收盤價；程式不會默默自動降級 feed。

## 3. 每晚更新如何運作

預設設定：

```dotenv
UPDATE_TIMEZONE=America/Chicago
UPDATE_HOUR=22
UPDATE_MINUTE=0
```

即芝加哥時間每天晚上 10 點，處理「昨天」，不是當天尚未完成的 session。時區會處理日光節約時間。修改後重新建立 scheduler 容器即可。

| 商品 | 來源 | 更新範圍 |
|---|---|---|
| `CSUS.L` | yfinance | Close 與 Adj Close |
| `initial_daily_prices.csv` 其餘 1,027 個 ticker，包含基金 ETF 與四個因子 ETF | Alpaca historical stock bars | raw Close 與 `adjustment=all` 的調整價 |
| 價格檔中沒有的 105 個 benchmark ticker | 不在本次更新名單 | 保留持倉，不補造價格 |

為讓原有風險與因子計算延續，**不能只新增 Close、把 Adj Close 留在舊日期，也不能直接把 Alpaca 的調整價接到 Yahoo 調整價尾端**。因此每次會重新讀取從初始價格起日到昨天的兩種價格歷史；這比只抓一天多一些請求，但可以同步公司行動造成的歷史調整。

每個 ticker／價格類型必須涵蓋目前分析中已使用的全部日期，且到達應有的最新交易日，才會一次發布整段新來源歷史。資料不完整時保留前一段完整序列，將狀態記錄為 `partial`，不把缺漏補成零，也不拿 Close 假裝 Adj Close。兩種價格分開驗證，因此某檔 Close 可能已更新、調整價仍保留舊來源；報告會分別列出。

更新程式會處理：

- Alpaca 分批請求與所有 `next_page_token`，避免只拿到第一頁。
- 429、暫時網路或伺服器錯誤的有界重試。
- 紐約交易日期、yfinance 的 end 日期不包含，以及美國／英國的休市日。
- 停機後的缺漏補抓；休市日不會造一筆收盤價。
- 完整交易提交、失敗回滾、唯一鍵 upsert，重跑不會重複價格。
- PostgreSQL advisory lock，避免手動更新與排程同時寫入。
- 排程服務重啟時補抓；不完整或失敗時最多另外重試兩次，間隔 30 分鐘。

排程獨立於瀏覽器，只要電腦、Docker 與 scheduler 持續運行就能執行。電腦休眠／關機時無法更新；重新啟動後會補抓。尚未填 key 時，網站可用，排程會顯示等待 key。

查看結果：

```powershell
docker compose logs --tail=80 scheduler
docker compose run --rm worker python verify_data.py
```

詳細個股狀態在 `reports/update_latest.json` 及資料庫的 `ops.update_symbol_status`。如果是 `partial`，查看 `incomplete_tickers`、各價格類型缺漏日期數及錯誤類別。

每天只更新資料庫；**原本的三份 CSV 保留不變**。如果需要對照 API 名稱，可在 `data/symbol_aliases.json` 加入明確的一對一 ticker 對照。

## 4. Dashboard 如何跟著更新

每次發布價格會在同一個交易中增加 `ops.data_revision`。頁面每 60 秒檢查這個小型版本值，版本改變才重新讀取資料並重算；快取也以資料版本為 key。原有的頁面配置、CSS、圖表類型與操作選項保留。

若部分股票還沒有新價格，歸因會採用基金與 benchmark 都有完整端點的最近共同期間，而不是將不同日期的報酬直接比較。資料來源與實際日期可透過 `verify_data.py`／更新報告查閱。

`home.py`、`fac_docs.py`、`portfolio_docs.py`、兩份文件 Markdown 及 LinkedIn 連結逐位元保留。`preservation_manifest.json` 記錄原始檔與 CSS 的校驗值。首頁圖片由原專案複製，未提供的原始 logo 沒有經過視覺比對。

四個計算檔保留 Monte Carlo、VaR／CVaR、Sharpe／Sortino、因子回歸、Treemap、Brinson、Radar、風險圖及相關係數熱圖。調整集中在資料庫查詢、按日期對齊、共同交易區間、快取及矩陣運算。模擬日期預設隨可用資料更新；計算沿用上一版已修正的期間利率與 downside deviation 定義，不新增介面控制項。

## 5. 初始資料與 benchmark 的範圍

| 項目 | 數量 |
|---|---:|
| 原始基金持倉 | 45 |
| 原始 benchmark 持倉 | 1,123 |
| 商品主檔（含未有價格的持倉） | 1,133 |
| 價格更新名單 | 1,028 |
| 原始寬表價格 | 281,569 |
| 初始核心價格紀錄（Close、Adj Close 分開） | 563,138 |
| 有價格可供分析的 benchmark 持倉 | 1,018 |
| 有價格的 benchmark 原始權重 | 94.332% |

三份附帶 CSV 完整保留你的內容。`core.holdings` 儲存 1,123 檔 benchmark 原始權重，合計 100%；分析用 1,018 檔的權重才按比例正規化至 100%。缺價格的 105 檔占 5.668%，明細在 `reports/benchmark_without_prices.csv`。因此畫面比較的是可定價的 proxy，不是完整的 1,123 檔。

本次兩份持倉共有 40 個 ticker，產業分類全部一致。benchmark 沒有 Industry 欄位的地方保留未知，不自行猜測。持倉日期未提供，仍為 NULL。價格更新不會自動追蹤基金成交、股數或資金流變化；這些仍需更新持倉資料。

## 6. 資料讀取介面

```python
from data_access import DataAccess

db = DataAccess()
fund = db.get_holdings("fund")
benchmark = db.get_holdings("r2500g_proxy")
benchmark_raw = db.get_holdings("r2500g_proxy", priced_only=False)
prices = db.get_prices(["ADUS", "IWO"], "2026-01-01", "2026-11-05")
close = db.get_prices(["ADUS", "IWO"], price_type="close")
```

預設讀調整價；日期上下限都包含。回傳有排序的 DatetimeIndex，欄位按指定 ticker 順序。價格不會自動向前／向後填值；來源保存在 `prices.attrs['sources']`。`Weight (%)` 是百分比，資料庫內部 `weight` 則是 0～1。

查詢最新分析用價格：

```powershell
docker compose exec db psql -U carlson -d carlson_fund -P pager=off -c "SELECT ticker,trade_date,price_type,price,source FROM mart.v_current_prices ORDER BY trade_date DESC,ticker,price_type LIMIT 10;"
```

## 7. 常用指令

```powershell
# 停止服務，保留資料
docker compose stop

# 重新啟動原有服務
docker compose up -d db app scheduler

# 查看服務狀態
docker compose ps

# 修改三份 CSV 後，重新驗證／匯入，再啟動
python start.py
```

不要使用 `docker compose down -v` 作為一般重啟指令，因為它會刪除本包的資料庫 volume。

## 8. 測試範圍

`validation.json` 是這份執行包的實際驗證紀錄。測試包含實際 PostgreSQL、全部原始價格比對、完整匯入與重跑、三個分析頁面、原始首頁與文件路由、公式加總、API 分頁／重試、整段歷史更新、交易回滾及開啟中的 Dashboard 重新讀取新版本。

外部 API 以可重現回應及原始資料重播測試；沒有使用你的 Alpaca 真實 key。Docker Desktop 與你的 Windows 環境需要由你執行上述啟動和手動更新指令確認。首頁測試使用暫時圖片 fixture 驗證原始檔案路徑，fixture 不在本包內；原始 logo 仍需由你原專案複製。

若要在另備的環境重跑純計算／provider 契約測試：

```powershell
python -m pip install -r requirements_test.txt
python -m pytest tests/test_analytics.py tests/test_providers.py tests/test_preservation.py -q
```

整合測試只適用另外準備的測試資料庫；請勿指向工作用資料庫。部署 AWS 前，本包先維持 localhost 綁定與獨立 volume，不會建立或變更 AWS 資源。

官方參考：[Alpaca historical bars](https://docs.alpaca.markets/us/reference/stockbars)、[Alpaca feed／權限說明](https://docs.alpaca.markets/us/v1.1/docs/market-data-faq)、[yfinance download](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)、[Streamlit fragment](https://docs.streamlit.io/develop/api-reference/execution-flow/st.fragment)、[APScheduler 時區排程](https://apscheduler.readthedocs.io/en/3.x/modules/triggers/cron.html)。
