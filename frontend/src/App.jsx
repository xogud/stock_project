import { useCallback, useEffect, useMemo, useState } from 'react'
import AccountPanel from './AccountPanel'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const VIEWED_STOCKS_KEY = 'stock-risk-viewed-stocks'
const MAX_VIEWED_STOCKS = 8

const levelClass = {
  NORMAL: 'normal',
  CAUTION: 'caution',
  DANGER: 'danger',
}

function formatPercent(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${(Number(value) * 100).toFixed(digits)}%`
}

function formatNumber(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return Number(value).toLocaleString('ko-KR')
}

function formatPrice(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-'
  return `${Number(value).toLocaleString('ko-KR', { maximumFractionDigits: 0 })}원`
}

async function fetchJson(path) {
  const response = await fetch(`${API_BASE}${path}`)
  if (!response.ok) {
    const message = await response.text()
    throw new Error(message || `API 요청 실패: ${response.status}`)
  }
  return response.json()
}

function loadViewedStocks() {
  if (typeof window === 'undefined') return []
  try {
    const parsed = JSON.parse(window.localStorage.getItem(VIEWED_STOCKS_KEY) || '[]')
    return Array.isArray(parsed) ? parsed.slice(0, MAX_VIEWED_STOCKS) : []
  } catch {
    return []
  }
}

function saveViewedStocks(items) {
  if (typeof window === 'undefined') return
  window.localStorage.setItem(VIEWED_STOCKS_KEY, JSON.stringify(items.slice(0, MAX_VIEWED_STOCKS)))
}

function MiniLineChart({ data, field, color, suffix = '' }) {
  const points = useMemo(() => {
    const values = data.map((item) => Number(item[field])).filter((value) => Number.isFinite(value))
    if (values.length < 2) return ''
    const min = Math.min(...values)
    const max = Math.max(...values)
    const range = max - min || 1
    return data
      .map((item, index) => {
        const value = Number(item[field])
        const x = (index / Math.max(data.length - 1, 1)) * 100
        const y = 56 - ((value - min) / range) * 48
        return `${x},${y}`
      })
      .join(' ')
  }, [data, field])

  const latest = data.length ? data[data.length - 1][field] : null

  return (
    <div className="chart">
      <svg viewBox="0 0 100 64" preserveAspectRatio="none" aria-hidden="true">
        <polyline className="grid-line" points="0,56 100,56" />
        <polyline className="grid-line" points="0,8 100,8" />
        <polyline className="line" points={points} style={{ stroke: color }} />
      </svg>
      <div className="chart-value">
        {suffix === '%' ? formatPercent(latest) : formatPrice(latest)}
      </div>
    </div>
  )
}

function RiskHelpTooltip() {
  return (
    <span className="risk-help-wrap">
      <button className="risk-help" type="button" aria-label="위험 확률 설명">
        !
      </button>
      <span className="risk-tooltip" role="tooltip">
        위험 확률은 실제 하락률이나 수익률 보장이 아니라, 과거 급락 조건과 현재 흐름이 얼마나 비슷한지 나타내는 위험 점수입니다. 매수·매도 추천이 아니라 확인용 지표입니다.
      </span>
    </span>
  )
}

function SearchPanel({ onSelect, onHome, viewedStocks = [], onClearViewedStocks, selected }) {
  const [query, setQuery] = useState('')
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    const timer = window.setTimeout(async () => {
      setLoading(true)
      try {
        const data = await fetchJson(`/api/stocks?q=${encodeURIComponent(query)}&limit=12`)
        setItems(data.items || [])
      } catch {
        setItems([])
      } finally {
        setLoading(false)
      }
    }, 180)
    return () => window.clearTimeout(timer)
  }, [query])

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div>
          <p className="eyebrow">종목 검색</p>
          <h1>단기 급락 위험 판단</h1>
        </div>
        <button className="home-button" type="button" onClick={onHome} aria-label="처음 페이지로 이동" title="처음 페이지">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M3 10.5 12 3l9 7.5" />
            <path d="M5 9.5V21h14V9.5" />
            <path d="M9 21v-7h6v7" />
          </svg>
        </button>
      </div>

      <div className="search-area">
        <label className="search-label" htmlFor="stock-search">종목 검색</label>
        <div className="search-box">
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <circle cx="11" cy="11" r="7" />
            <path d="m16.5 16.5 4 4" />
          </svg>
          <input
            id="stock-search"
            className="search-input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="종목명 또는 코드 입력 예: 삼성전자, 005930"
          />
        </div>

        {(query || loading) && (
          <div className="result-list">
            {loading && <div className="muted">검색 중</div>}
            {!loading && items.map((item) => (
              <button key={item.ticker} className="stock-row" onClick={() => onSelect(item)}>
                <span>
                  <strong>{item.name}</strong>
                  <small>{item.code} · {item.market}{item.sector ? ` · ${item.sector}` : ''}</small>
                </span>
                <span className="ticker">{item.ticker}</span>
              </button>
            ))}
            {!loading && items.length === 0 && <div className="muted">검색 결과가 없습니다.</div>}
          </div>
        )}
      </div>

      <AccountPanel selected={selected} onSelect={onSelect} />

      {viewedStocks.length > 0 && (
        <div className="viewed-section">
          <div className="section-head">
            <strong>최근 확인한 종목</strong>
            <button type="button" onClick={onClearViewedStocks}>지우기</button>
          </div>
          <div className="viewed-list">
            {viewedStocks.map((item) => (
              <button key={item.ticker} className="viewed-row" type="button" onClick={() => onSelect(item)}>
                <span>
                  <strong>{item.name}</strong>
                  <small>{item.code} · {item.market}{item.sector ? ` · ${item.sector}` : ''}</small>
                </span>
                <span className="ticker">{item.ticker}</span>
              </button>
            ))}
          </div>
        </div>
      )}

    </aside>
  )
}

function RiskHeader({ prediction }) {
  const latest = prediction.latest
  const className = levelClass[latest.risk_level] || 'normal'
  return (
    <section className={`risk-hero ${className}`}>
      <div>
        <p className="eyebrow">{prediction.stock.name} · {prediction.stock.code}</p>
        <h2>{latest.risk_level_ko}</h2>
        <p>{latest.description}</p>
      </div>
      <div className="probability">
        <span>{formatPercent(latest.risk_probability)}</span>
        <small>단기 급락 가능성</small>
      </div>
    </section>
  )
}

function MetricGrid({ prediction }) {
  const latest = prediction.latest
  return (
    <section className="metric-grid">
      <div className="metric">
        <span>최근 기준일</span>
        <strong>{latest.date}</strong>
      </div>
      <div className="metric">
        <span>최근 종가</span>
        <strong>{formatPrice(latest.close)}</strong>
      </div>
      <div className="metric">
        <span>거래량</span>
        <strong>{formatNumber(latest.volume)}</strong>
      </div>
      <div className="metric">
        <span>최근 1일 수익률</span>
        <strong>{formatPercent(latest.return_1d, 2)}</strong>
      </div>
    </section>
  )
}

function BacktestSummary({ data }) {
  if (!data?.available) return null
  const overall = data.overall
  return (
    <section className="panel compact">
      <div className="panel-title">
        <h3>최근 백테스트</h3>
        <span>{overall.first_prediction_date} ~ {overall.last_comparable_date}</span>
      </div>
      <div className="score-row">
        <div>
          <strong>{formatPercent(overall.f1)}</strong>
          <span>균형 점수</span>
          <small>위험 포착과 오경보를 함께 본 점수</small>
        </div>
        <div>
          <strong>{formatPercent(overall.precision)}</strong>
          <span>위험 판단 적중률</span>
          <small>위험이라고 한 종목 중 실제 급락한 비율</small>
        </div>
        <div>
          <strong>{formatPercent(overall.recall)}</strong>
          <span>급락 포착률</span>
          <small>실제 급락 종목 중 시스템이 잡아낸 비율</small>
        </div>
        <div>
          <strong>{formatPercent(overall.top10_mean_hit_rate)}</strong>
          <span>Top10 적중률</span>
          <small>위험 상위 10개 안에서 맞춘 비율</small>
        </div>
      </div>
    </section>
  )
}

function MarketContextCard({ context }) {
  if (!context?.available) return (
    <section className="panel market-context">
      <div className="panel-title"><h3>시장 동반 변동</h3><span>{context?.date || ''}</span></div>
      <p className="muted">해당 기준일의 시장 비교 데이터가 없습니다.</p>
    </section>
  )
  const index = context.index
  const sector = context.sector
  return (
    <section className="panel market-context">
      <div className="panel-title">
        <h3>시장 동반 변동</h3>
        <span>{context.date}</span>
      </div>
      <div className="market-context-body">
        <div>
          <strong>{context.label}</strong>
          <p>{context.message}</p>
        </div>
        <div className="market-context-metrics">
          <span>
            하락 종목
            <strong>{formatPercent(context.down_ratio)}</strong>
            <small>{formatNumber(context.down_count)} / {formatNumber(context.total_count)}</small>
          </span>
          <span>
            -3% 이상 하락
            <strong>{formatPercent(context.sharp_drop_ratio)}</strong>
            <small>{formatNumber(context.sharp_drop_count)}개 종목</small>
          </span>
          <span>
            평균 수익률
            <strong>{formatPercent(context.market_mean_return, 2)}</strong>
            <small>같은 날 전체 종목 기준</small>
          </span>
          <span>
            종합주가지수
            <strong>{index?.available ? formatPercent(index.return_1d, 2) : '-'}</strong>
            <small>{index?.available ? `5일 ${formatPercent(index.return_5d, 2)}` : '데이터 없음'}</small>
          </span>
          {sector && (
            <span>
              섹터 하락
              <strong>{formatPercent(sector.down_ratio)}</strong>
              <small>{sector.name} · {formatNumber(sector.down_count)} / {formatNumber(sector.total_count)}</small>
            </span>
          )}
          {sector && (
            <span>
              섹터 평균
              <strong>{formatPercent(sector.mean_return, 2)}</strong>
              <small>-3% 이상 {formatPercent(sector.sharp_drop_ratio)}</small>
            </span>
          )}
          {index?.relative_return_1d !== null && index?.relative_return_1d !== undefined && (
            <span>
              지수 대비 차이
              <strong>{formatPercent(index.relative_return_1d, 2)}</strong>
              <small>종목 수익률 - 종합주가지수</small>
            </span>
          )}
        </div>
      </div>
    </section>
  )
}

function DisclosureCard({ stockCode, riskProbability }) {
  const [disclosures, setDisclosures] = useState(null)
  const [loading, setLoading] = useState(true)
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    const controller = new AbortController()
    let active = true
    setLoading(true)
    setDisclosures(null)
    const params = new URLSearchParams()
    if (Number.isFinite(riskProbability)) params.set('risk_probability', String(riskProbability))
    fetch(`${API_BASE}/api/disclosures/${encodeURIComponent(stockCode)}?${params}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error('공시 조회에 실패했습니다. 다시 시도해 주세요.')
        return response.json()
      })
      .then((data) => { if (active) setDisclosures(data) })
      .catch((error) => {
        if (active) setDisclosures({ available: false, reason: error.message, items: [] })
      })
      .finally(() => { if (active) setLoading(false) })
    return () => { active = false; controller.abort() }
  }, [stockCode, riskProbability, attempt])
  const summary = disclosures?.summary
  return (
    <section className="panel disclosure-panel">
      <div className="panel-title">
        <h3>최근 공시</h3>
        <span>{disclosures?.period ? `${disclosures.period.from} ~ ${disclosures.period.to}` : 'DART'}</span>
      </div>
      {loading && <p className="muted" role="status">공시를 불러오는 중입니다.</p>}
      {!loading && !disclosures?.available && (
        <div>
        <p className="muted">{disclosures?.reason || '공시 데이터를 불러오지 못했습니다.'}</p>
        <button type="button" onClick={() => setAttempt((value) => value + 1)}>다시 시도</button>
        </div>
      )}
      {disclosures?.available && disclosures.items.length === 0 && (
        <p className="muted">최근 조회 기간에 등록된 공시가 없습니다.</p>
      )}
      {disclosures?.available && summary && (
        <div className="disclosure-summary-grid">
          <div>
            <span>공시 위험</span>
            <strong>{summary.risk_level}</strong>
            <small>점수 {summary.risk_score} · 분석 {summary.analyzed_count}건</small>
          </div>
          <div>
            <span>주의 공시</span>
            <strong>{summary.caution_count}</strong>
            <small>최근 공시 {summary.total_count}건 중</small>
          </div>
          <div>
            <span>주요 키워드</span>
            <strong>{summary.top_keywords?.length ? summary.top_keywords.slice(0, 2).join(', ') : '-'}</strong>
            <small>{summary.top_keywords?.slice(2).join(', ') || '추가 키워드 없음'}</small>
          </div>
        </div>
      )}
      {disclosures?.available && summary?.interpretation && (
        <p className="disclosure-interpretation">{summary.interpretation}</p>
      )}
      {disclosures?.available && disclosures.items.length > 0 && (
        <div className="disclosure-list">
          {disclosures.items.map((item) => (
            <a className="disclosure-row" href={item.url || '#'} target="_blank" rel="noreferrer" key={item.receipt_no}>
              <span>
                <strong>{item.report_name}</strong>
                <small>{item.date} · {item.submitter || item.corp_name}</small>
                {item.analysis?.summary && (
                  <small className="disclosure-summary">{item.analysis.summary}</small>
                )}
                {item.analysis?.snippets?.length > 0 && (
                  <small className="disclosure-snippet">
                    {item.analysis.snippets[0].category}: {item.analysis.snippets[0].text}
                  </small>
                )}
              </span>
              {item.tags?.length > 0 && (
                <em>{item.tags.join(', ')}</em>
              )}
            </a>
          ))}
        </div>
      )}
    </section>
  )
}

function DashboardOverview({ backtest, topRisks }) {
  const overall = backtest?.available ? backtest.overall : null
  const leadRisk = topRisks?.items?.[0]
  const marketContext = topRisks?.market_context

  return (
    <section className="dashboard-hero">
      <div className="hero-copy">
        <div className="hero-kicker">
          <p className="eyebrow">Stock Risk Monitor</p>
          <RiskHelpTooltip />
        </div>
        <h2>최근 급락 위험 대시보드</h2>
      </div>
      <div className="hero-stats">
        <div>
          <span>상위 위험 종목</span>
          <strong>{leadRisk?.stock_name || '-'}</strong>
          <small>{leadRisk ? formatPercent(leadRisk.risk_probability) : '-'}</small>
        </div>
        <div>
          <span>Top10 적중률</span>
          <strong>{overall ? formatPercent(overall.top10_mean_hit_rate) : '-'}</strong>
          <small>위험 상위 10개 안에서 맞춘 비율</small>
        </div>
        <div>
          <span>균형 점수</span>
          <strong>{overall ? formatPercent(overall.f1) : '-'}</strong>
          <small>포착률과 적중률의 균형</small>
        </div>
        <div>
          <span>시장 동반 변동</span>
          <strong>{marketContext?.label || '-'}</strong>
          <small>
            {marketContext?.index?.available
              ? `KOSPI ${formatPercent(marketContext.index.return_1d, 2)}`
              : marketContext?.available ? `하락 종목 ${formatPercent(marketContext.down_ratio)}` : '-'}
          </small>
        </div>
      </div>
    </section>
  )
}

function TopRisks({ data, onSelect }) {
  if (!data?.items?.length) return null
  return (
    <section className="panel">
      <div className="panel-title">
        <h3>최근 위험 상위 종목</h3>
        <span>{data.latest_date}</span>
      </div>
      <div className="top-list">
        {data.items.slice(0, 6).map((item, index) => (
          <button key={item.ticker} className="top-row" onClick={() => onSelect({
            ticker: item.ticker,
            name: item.stock_name,
            code: item.stock_code,
            market: item.market,
            sector: item.sector,
          })}>
            <span className="rank">{index + 1}</span>
            <span className="top-name">
              <strong>{item.stock_name}</strong>
              <small>{item.stock_code} · {item.market}{item.sector ? ` · ${item.sector}` : ''}</small>
            </span>
            <span className={`badge ${levelClass[item.risk_level]}`}>{item.risk_level_ko}</span>
            <span className="top-prob">{formatPercent(item.risk_probability)}</span>
          </button>
        ))}
      </div>
    </section>
  )
}

function DashboardBottom({ backtest, topRisks }) {
  const items = topRisks?.items || []
  const overall = backtest?.available ? backtest.overall : null
  const sectorRows = Object.values(items.reduce((acc, item) => {
    const sector = item.sector || '섹터 미분류'
    if (!acc[sector]) {
      acc[sector] = {
        sector,
        count: 0,
        totalRisk: 0,
        maxRisk: 0,
        leadName: item.stock_name,
      }
    }
    acc[sector].count += 1
    acc[sector].totalRisk += Number(item.risk_probability || 0)
    if (Number(item.risk_probability || 0) >= acc[sector].maxRisk) {
      acc[sector].maxRisk = Number(item.risk_probability || 0)
      acc[sector].leadName = item.stock_name
    }
    return acc
  }, {}))
    .map((item) => ({
      ...item,
      averageRisk: item.count ? item.totalRisk / item.count : 0,
    }))
    .sort((a, b) => b.averageRisk - a.averageRisk)
    .slice(0, 5)

  const precision = overall?.precision ?? null
  const recall = overall?.recall ?? null
  const f1 = overall?.f1 ?? null
  const top10 = overall?.top10_mean_hit_rate ?? null

  return (
    <section className="dashboard-bottom">
      <div className="panel sector-panel">
        <div className="panel-title">
          <h3>섹터별 위험 분포</h3>
          <span>상위 위험 종목 기준</span>
        </div>
        {sectorRows.length === 0 && <p className="muted">표시할 섹터 데이터가 없습니다.</p>}
        {sectorRows.length > 0 && (
          <div className="sector-list">
            {sectorRows.map((item) => (
              <div className="sector-row" key={item.sector}>
                <div className="sector-row-head">
                  <strong>{item.sector}</strong>
                  <span>{item.count}개</span>
                </div>
                <div className="sector-bar" aria-hidden="true">
                  <span style={{ width: `${Math.min(item.averageRisk * 100, 100)}%` }} />
                </div>
                <small>
                  평균 위험 {formatPercent(item.averageRisk)} · 최고 위험 {item.leadName} {formatPercent(item.maxRisk)}
                </small>
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="panel operation-panel">
        <div className="panel-title">
          <h3>판단 보조 요약</h3>
          <span>사용자가 봐야 할 포인트</span>
        </div>
        <div className="operation-grid">
          <div>
            <strong>{top10 !== null ? formatPercent(top10) : '-'}</strong>
            <span>위험 상위 10개 안에서 실제 급락을 맞춘 비율</span>
          </div>
          <div>
            <strong>{f1 !== null ? formatPercent(f1) : '-'}</strong>
            <span>위험 포착률과 적중률을 같이 본 균형 점수</span>
          </div>
          <div>
            <strong>{precision !== null ? formatPercent(precision) : '-'}</strong>
            <span>위험이라고 판단한 종목 중 실제 급락 비율</span>
          </div>
          <div>
            <strong>{recall !== null ? formatPercent(recall) : '-'}</strong>
            <span>실제 급락 종목을 시스템이 잡아낸 비율</span>
          </div>
        </div>
        <p className="operation-note">
          이 화면은 매수·매도 추천이 아니라, 어떤 종목과 섹터에 단기 급락 위험 신호가 몰려 있는지 확인하는 용도입니다.
        </p>
      </div>
    </section>
  )
}

function PredictionView({ prediction }) {
  const latest = prediction.latest
  const topReasons = latest.reasons.slice(0, 3)
  return (
    <main className="content">
      <RiskHeader prediction={prediction} />
      <MetricGrid prediction={prediction} />
      <MarketContextCard context={latest.market_context} />
      <DisclosureCard key={prediction.stock.code} stockCode={prediction.stock.code} riskProbability={latest.risk_probability} />

      <section className="panel insight-panel">
        <div className="panel-title">
          <h3>핵심 위험 신호</h3>
          <span>위험 판단 이유 TOP 3</span>
        </div>
        <div className="insight-grid">
          {topReasons.map((item, index) => (
            <div className="insight-card" key={item.feature}>
              <span className="insight-rank">{index + 1}</span>
              <div>
                <strong>{item.label}</strong>
                <em>{item.value}</em>
                {item.summary && <small>{item.summary}</small>}
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="split">
        <div className="panel">
          <div className="panel-title">
            <h3>판단 이유</h3>
            <span>쉽게 풀어쓴 설명</span>
          </div>
          <p className="explain">{latest.explanation}</p>
          <div className="reason-list">
            {latest.reasons.map((item) => (
              <div className="reason" key={item.feature}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                {item.summary && <small>{item.summary}</small>}
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-title">
            <h3>해석</h3>
            <span>사용자 관점</span>
          </div>
          <p className="interpretation">{latest.description}</p>
          <p className="muted">
            위험 확률은 주가가 그 확률만큼 하락한다는 뜻이 아니라, 과거 급락 조건과
            유사한 정도를 나타냅니다.
          </p>
          {latest.market_context?.available && (
            <p className="muted">
              {latest.market_context.message}
            </p>
          )}
        </div>
      </section>

      <section className="split charts">
        <div className="panel">
          <div className="panel-title">
            <h3>주가 흐름</h3>
            <span>최근 {prediction.series.length}거래일</span>
          </div>
          <MiniLineChart data={prediction.series} field="close" color="#2563eb" />
        </div>

        <div className="panel">
          <div className="panel-title">
            <h3>위험 가능성 흐름</h3>
            <span>0~100%</span>
          </div>
          <MiniLineChart data={prediction.series} field="risk_probability" color="#dc2626" suffix="%" />
        </div>
      </section>

      <section className="panel">
        <div className="panel-title">
          <h3>최근 판단 내역</h3>
          <span>최근 15건</span>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>날짜</th>
                <th>종가</th>
                <th>거래량</th>
                <th>위험 확률</th>
                <th>단계</th>
              </tr>
            </thead>
            <tbody>
              {prediction.recent.map((row) => (
                <tr key={row.date}>
                  <td>{row.date}</td>
                  <td>{formatPrice(row.close)}</td>
                  <td>{formatNumber(row.volume)}</td>
                  <td>{formatPercent(row.risk_probability)}</td>
                  <td><span className={`badge ${levelClass[row.risk_level]}`}>{row.risk_level_ko}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  )
}

export default function App() {
  const [selected, setSelected] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [backtest, setBacktest] = useState(null)
  const [topRisks, setTopRisks] = useState(null)
  const [viewedStocks, setViewedStocks] = useState(loadViewedStocks)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSelectStock = useCallback((stock) => {
    setSelected(stock)
    setViewedStocks((previous) => {
      const normalized = {
        ticker: stock.ticker,
        name: stock.name || stock.stock_name,
        code: stock.code || stock.stock_code,
        market: stock.market,
        sector: stock.sector,
      }
      const next = [
        normalized,
        ...previous.filter((item) => item.ticker !== normalized.ticker),
      ].slice(0, MAX_VIEWED_STOCKS)
      saveViewedStocks(next)
      return next
    })
  }, [])

  const clearViewedStocks = useCallback(() => {
    setViewedStocks([])
    saveViewedStocks([])
  }, [])

  const refreshDashboard = useCallback(() => {
    fetchJson('/api/backtest-summary').then(setBacktest).catch(() => setBacktest(null))
    fetchJson('/api/top-risks?limit=10').then(setTopRisks).catch(() => setTopRisks(null))
  }, [])

  useEffect(() => {
    refreshDashboard()
    const timer = window.setInterval(refreshDashboard, 60000)
    return () => window.clearInterval(timer)
  }, [refreshDashboard])

  useEffect(() => {
    if (!selected) return
    let active = true
    setLoading(true)
    setError('')
    fetchJson(`/api/predict/${encodeURIComponent(selected.ticker)}?days=365`)
      .then((data) => {
        if (active) setPrediction(data)
      })
      .catch((err) => {
        if (!active) return
        setPrediction(null)
        setError(err.message)
      })
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [selected])

  const goHome = useCallback(() => {
    setSelected(null)
    setPrediction(null)
    setLoading(false)
    setError('')
    refreshDashboard()
  }, [refreshDashboard])

  return (
    <div className="app-shell">
      <SearchPanel
        onSelect={handleSelectStock}
        onHome={goHome}
        viewedStocks={viewedStocks}
        onClearViewedStocks={clearViewedStocks}
        selected={selected}
      />
      <div className="main-area">
        {loading && <div className="empty-state">분석 중입니다.</div>}
        {error && <div className="error-state">분석에 실패했습니다. {error}</div>}
        {!loading && !prediction && !error && (
          <main className="content">
            <DashboardOverview backtest={backtest} topRisks={topRisks} />
            <section className="dashboard-grid">
              <div className="dashboard-main">
                <MarketContextCard context={topRisks?.market_context} />
                <BacktestSummary data={backtest} />
              </div>
              <TopRisks data={topRisks} onSelect={handleSelectStock} />
            </section>
            <DashboardBottom backtest={backtest} topRisks={topRisks} />
          </main>
        )}
        {!loading && prediction && <PredictionView prediction={prediction} />}
      </div>
    </div>
  )
}
