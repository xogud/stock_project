CREATE DATABASE IF NOT EXISTS stock_risk_project
  DEFAULT CHARACTER SET utf8mb4
  DEFAULT COLLATE utf8mb4_0900_ai_ci;

USE stock_risk_project;

CREATE TABLE stock_info (
  stock_code VARCHAR(10) NOT NULL,
  ticker VARCHAR(20) NOT NULL,
  stock_name VARCHAR(100) NOT NULL,
  market VARCHAR(20) NOT NULL,
  sector VARCHAR(50) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (stock_code),
  UNIQUE KEY uq_stock_info_ticker (ticker),
  KEY idx_stock_info_market_sector (market, sector)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE stock_price (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  open_price DECIMAL(18, 4) NOT NULL,
  high_price DECIMAL(18, 4) NOT NULL,
  low_price DECIMAL(18, 4) NOT NULL,
  close_price DECIMAL(18, 4) NOT NULL,
  adj_close_price DECIMAL(18, 4) NOT NULL,
  volume BIGINT NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_stock_price_code_date (stock_code, trade_date),
  KEY idx_stock_price_date (trade_date),
  CONSTRAINT fk_stock_price_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE feature_data (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  return_1d DECIMAL(18, 10) NULL,
  ma20 DECIMAL(18, 4) NULL,
  ma20_gap DECIMAL(18, 10) NULL,
  volatility_20d DECIMAL(18, 10) NULL,
  volume_change_1d DECIMAL(18, 10) NULL,
  volume_ratio_20d DECIMAL(18, 10) NULL,
  rsi_14 DECIMAL(18, 10) NULL,
  macd DECIMAL(18, 10) NULL,
  macd_signal DECIMAL(18, 10) NULL,
  macd_hist DECIMAL(18, 10) NULL,
  bollinger_upper DECIMAL(18, 4) NULL,
  bollinger_lower DECIMAL(18, 4) NULL,
  bollinger_width DECIMAL(18, 10) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_feature_data_code_date (stock_code, trade_date),
  KEY idx_feature_data_date (trade_date),
  CONSTRAINT fk_feature_data_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE anomaly_result (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  z_score DECIMAL(18, 10) NULL,
  ewma_value DECIMAL(18, 10) NULL,
  cusum_pos DECIMAL(18, 10) NULL,
  cusum_neg DECIMAL(18, 10) NULL,
  anomaly_score DECIMAL(18, 10) NULL,
  is_anomaly TINYINT(1) NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_anomaly_result_code_date (stock_code, trade_date),
  KEY idx_anomaly_result_date (trade_date),
  KEY idx_anomaly_result_is_anomaly (is_anomaly),
  CONSTRAINT fk_anomaly_result_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE risk_label (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  target_window_days INT NOT NULL DEFAULT 5,
  risk_label TINYINT NOT NULL,
  label_reason VARCHAR(255) NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_risk_label_code_date_window (stock_code, trade_date, target_window_days),
  KEY idx_risk_label_label (risk_label),
  CONSTRAINT fk_risk_label_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT,
  CONSTRAINT chk_risk_label_value CHECK (risk_label IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE model_info (
  id BIGINT NOT NULL AUTO_INCREMENT,
  model_name VARCHAR(100) NOT NULL,
  model_type VARCHAR(50) NOT NULL,
  model_version VARCHAR(50) NOT NULL,
  target_window_days INT NOT NULL DEFAULT 5,
  model_path VARCHAR(500) NOT NULL,
  feature_columns JSON NOT NULL,
  train_start_date DATE NULL,
  train_end_date DATE NULL,
  metrics JSON NULL,
  is_active TINYINT(1) NOT NULL DEFAULT 0,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_model_info_version (model_name, model_version),
  KEY idx_model_info_active (is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE prediction_result (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  model_id BIGINT NOT NULL,
  risk_probability DECIMAL(8, 6) NOT NULL,
  risk_level ENUM('NORMAL', 'CAUTION', 'DANGER') NOT NULL,
  target_window_days INT NOT NULL DEFAULT 5,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  UNIQUE KEY uq_prediction_code_date_model (stock_code, trade_date, model_id),
  KEY idx_prediction_date (trade_date),
  KEY idx_prediction_level (risk_level),
  CONSTRAINT fk_prediction_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT,
  CONSTRAINT fk_prediction_model
    FOREIGN KEY (model_id) REFERENCES model_info (id)
    ON UPDATE CASCADE
    ON DELETE RESTRICT,
  CONSTRAINT chk_prediction_probability CHECK (risk_probability >= 0 AND risk_probability <= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE alert_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  prediction_id BIGINT NULL,
  stock_code VARCHAR(10) NOT NULL,
  trade_date DATE NOT NULL,
  risk_level ENUM('NORMAL', 'CAUTION', 'DANGER') NOT NULL,
  message TEXT NOT NULL,
  top_factors JSON NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_alert_log_code_date (stock_code, trade_date),
  KEY idx_alert_log_level (risk_level),
  CONSTRAINT fk_alert_log_prediction
    FOREIGN KEY (prediction_id) REFERENCES prediction_result (id)
    ON UPDATE CASCADE
    ON DELETE SET NULL,
  CONSTRAINT fk_alert_log_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

CREATE TABLE user_log (
  id BIGINT NOT NULL AUTO_INCREMENT,
  stock_code VARCHAR(10) NOT NULL,
  start_date DATE NULL,
  end_date DATE NULL,
  action_type VARCHAR(50) NOT NULL,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (id),
  KEY idx_user_log_code_created (stock_code, created_at),
  CONSTRAINT fk_user_log_stock
    FOREIGN KEY (stock_code) REFERENCES stock_info (stock_code)
    ON UPDATE CASCADE
    ON DELETE RESTRICT
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
