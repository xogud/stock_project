# Stock Risk Warning Dashboard

국내 주식의 향후 5거래일 급락 위험을 분석하는 React + FastAPI 기반 대시보드입니다.

## 주요 기능

- KRX 상장 목록에서 기본 300개 종목을 자동 선정합니다.
- yfinance에서 일봉 가격 데이터를 수집하고, 실패 시 로컬 raw 캐시를 사용합니다.
- 가격, 거래량, 기술지표 feature와 이상탐지 feature를 생성합니다.
- 향후 5거래일 급락 여부를 라벨로 만들고 XGBoost 모델을 학습합니다.
- validation 기준으로 위험 threshold를 자동 선택합니다.
- walk-forward validation, top-N hit rate, 단계별 실제 위험률 등 실전형 지표를 저장합니다.
- React 화면에서 종목 검색, 위험 단계, 위험 확률, 판단 이유, 차트를 제공합니다.

## 실행 방법

백엔드 의존성 설치:

```powershell
python -m pip install -r requirements.txt
```

프론트엔드 의존성 설치:

```powershell
cd frontend
npm install
cd ..
```

데이터와 모델 생성:

```powershell
python run_pipeline.py --max-stocks 300
```

백엔드 실행:

```powershell
uvicorn app:app --reload --port 8000
```

프론트엔드 실행:

```powershell
cd frontend
npm run dev
```

브라우저에서 접속:

```text
http://localhost:5173
```

## 파이프라인

`run_pipeline.py`는 아래 순서로 전체 작업을 실행합니다.

```text
collect_and_preprocess_stocks.py
rebuild_feature_data.py
detect_anomalies.py
generate_risk_labels.py
train_xgboost_model.py
predict_risk.py
```

자주 쓰는 옵션:

```powershell
python run_pipeline.py --max-stocks 300
python run_pipeline.py --universe static
python run_pipeline.py --skip-collect
python run_pipeline.py --skip-train
python run_pipeline.py --allow-stale
```

## Threshold 비교

교수님 피드백처럼 여러 위험 기준값을 비교하려면 아래 스크립트를 실행합니다.

```powershell
python threshold_sweep_backtest.py
```

원하는 threshold만 지정할 수도 있습니다.

```powershell
python threshold_sweep_backtest.py --thresholds 0.75,0.8,0.82,0.85,0.9,0.95
```

결과 파일:

```text
data/processed/daily_backtest/threshold_sweep/threshold_summary_latest.csv
data/processed/daily_backtest/threshold_sweep/threshold_daily_latest.csv
data/processed/daily_backtest/threshold_sweep/threshold_report_latest.json
```

## API

- `GET /api/health`
- `GET /api/database/status`
- `GET /api/stocks?q=삼성전자`
- `GET /api/predict/{ticker}`
- `GET /api/top-risks?limit=10`
- `GET /api/backtest-summary`

## MySQL 연결

앱은 MySQL 연결 정보가 있으면 DB를 먼저 조회하고, 접속할 수 없으면 CSV 파일로 자동 전환합니다.

PowerShell 예시:

```powershell
$env:MYSQL_ENABLED="1"
$env:MYSQL_HOST="localhost"
$env:MYSQL_PORT="3306"
$env:MYSQL_USER="root"
$env:MYSQL_PASSWORD="본인_MySQL_비밀번호"
$env:MYSQL_DATABASE="stock_risk_project"
python -m uvicorn app:app --port 8000
```

CSV 산출물을 MySQL에 적재:

```powershell
python database/load_csv_to_mysql.py --init-schema --password "본인_MySQL_비밀번호"
```

DB 연결 상태 확인:

```text
http://127.0.0.1:8000/api/database/status
```

## 현재 날짜 기준 업데이트

최신 주가 수집, 예측 갱신, 백테스트 갱신, threshold 비교를 한 번에 실행합니다.

```powershell
python update_current_system.py --skip-train --load-mysql --mysql-password "본인_MySQL_비밀번호"
```

계속 자동 갱신하려면 반복 간격을 분 단위로 지정합니다.

```powershell
python update_current_system.py --skip-train --load-mysql --mysql-password "본인_MySQL_비밀번호" --repeat-minutes 1440
```

`daily_incremental_backtest.py`는 `--end-date`를 지정하지 않으면 실행 당일 날짜를 자동으로 사용합니다.

## 주요 산출물

- `data/processed/stock_price.csv`
- `data/processed/feature_data.csv`
- `data/processed/anomaly_result.csv`
- `data/processed/risk_label.csv`
- `data/processed/prediction_result.csv`
- `data/processed/alert_log.csv`
- `data/processed/pipeline_status.json`
- `models/xgboost_risk_model_latest.joblib`
- `data/model_metrics/xgboost_risk_model_latest_metrics.json`
- `data/model_metrics/xgboost_risk_model_latest_feature_importance.csv`
- `data/model_metrics/xgboost_risk_model_latest_feature_stats.json`

## 라벨 정의

`risk_label`은 향후 5거래일 기준 아래 조건 중 하나를 만족하면 `1`입니다.

- 향후 5거래일 누적수익률 `-8%` 이하
- 향후 5거래일 중 일간수익률 `-6%` 이하

미래 5거래일 데이터가 부족한 행은 `is_label_available=0`으로 표시하며 학습에서 제외합니다.

## 인코딩

- 소스 코드와 `data/processed` CSV는 UTF-8 또는 UTF-8-SIG 기준입니다.
- MySQL import용 `data/mysql_import_ready` CSV는 `cp949`로 저장합니다.
- PowerShell에서 한글이 깨져 보이면 파일 자체가 깨진 것이 아니라 콘솔 출력 인코딩 문제일 수 있습니다.

인코딩/깨진 문자열 점검:

```powershell
python validate_text_encoding.py
```
