FROM node:22-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MYSQL_ENABLED=0
ENV PORT=8000

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY *.py ./
COPY database ./database
COPY data/raw ./data/raw
COPY data/reference ./data/reference
COPY data/processed/stock_info.csv ./data/processed/stock_info.csv
COPY data/processed/feature_data.csv ./data/processed/feature_data.csv
COPY data/processed/market_index.csv ./data/processed/market_index.csv
COPY data/processed/prediction_result.csv ./data/processed/prediction_result.csv
COPY data/processed/dart_corp_codes.csv ./data/processed/dart_corp_codes.csv
COPY data/processed/daily_backtest/daily_backtest_report_latest.json ./data/processed/daily_backtest/daily_backtest_report_latest.json
COPY data/model_metrics/xgboost_risk_model_latest_feature_importance.csv ./data/model_metrics/xgboost_risk_model_latest_feature_importance.csv
COPY data/model_metrics/xgboost_risk_model_latest_feature_stats.json ./data/model_metrics/xgboost_risk_model_latest_feature_stats.json
COPY models/xgboost_risk_model_latest.joblib ./models/xgboost_risk_model_latest.joblib
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
