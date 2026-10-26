import { useCallback, useEffect, useRef, useState } from 'react'
import { LogIn, LogOut, Star, Trash2, UserRound, X } from 'lucide-react'
import './accounts.css'

const API_BASE = import.meta.env.VITE_API_BASE || ''

async function accountRequest(path, method = 'GET', body) {
  const response = await fetch(`${API_BASE}/api${path}`, {
    method, credentials: 'include', cache: 'no-store',
    headers: { 'Content-Type': 'application/json', 'X-Requested-With': 'stock-risk' },
    body: body ? JSON.stringify(body) : undefined,
  })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : '요청을 처리하지 못했습니다. 입력 내용과 연결을 확인해 주세요.')
    error.status = response.status
    throw error
  }
  return data
}

function LoginDialog({ onClose, onAuthenticated, firebaseConfig }) {
  const dialog = useRef(null)
  const [mode, setMode] = useState('login')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const registering = mode === 'register'
  useEffect(() => {
    const element = dialog.current
    element.showModal()
    return () => element.close()
  }, [])
  async function submit(event) {
    event.preventDefault()
    const values = new FormData(event.currentTarget)
    if (registering && values.get('password') !== values.get('confirm')) {
      setError('비밀번호 확인이 일치하지 않습니다.')
      return
    }
    setBusy(true)
    setError('')
    try {
      let data
      if (firebaseConfig) {
        const { firebaseLogin } = await import('./firebaseLogin')
        data = await firebaseLogin(firebaseConfig, registering, values.get('username'), values.get('password'),
          (idToken) => accountRequest('/auth/session', 'POST', { id_token: idToken }))
      } else {
        data = await accountRequest(`/auth/${mode}`, 'POST', {
          username: values.get('username'), password: values.get('password'),
        })
      }
      onAuthenticated(data.user)
      onClose()
    } catch (failure) { setError(failure.message) }
    finally { setBusy(false) }
  }
  return (
    <dialog className="account-dialog" ref={dialog} aria-labelledby="account-title" onCancel={(event) => { if (busy) event.preventDefault(); else onClose() }}>
      <div className="account-dialog-head">
        <h3 id="account-title">{registering ? '회원가입' : '로그인'}</h3>
        <button className="account-icon" type="button" onClick={onClose} disabled={busy} aria-label="닫기" title="닫기"><X size={18} /></button>
      </div>
      <div className="account-tabs" role="tablist" aria-label="계정">
        {['login', 'register'].map((tab) => <button type="button" role="tab" key={tab} aria-selected={mode === tab} disabled={busy}
          onClick={() => { setMode(tab); setError('') }}>{tab === 'login' ? '로그인' : '회원가입'}</button>)}
      </div>
      <form key={mode} onSubmit={submit} className="account-form">
        <label htmlFor="account-username">{firebaseConfig ? '이메일' : '아이디'}</label>
        <input id="account-username" name="username" type={firebaseConfig ? 'email' : 'text'} autoFocus autoComplete="username" required minLength={3} maxLength={firebaseConfig ? 254 : 32}
          pattern={firebaseConfig ? undefined : '[A-Za-z0-9_]+'} placeholder={firebaseConfig ? 'name@example.com' : '영문, 숫자, 밑줄 3~32자'} disabled={busy} />
        <label htmlFor="account-password">비밀번호</label>
        <input id="account-password" name="password" type="password" autoComplete={registering ? 'new-password' : 'current-password'}
          required minLength={10} maxLength={128} placeholder="10자 이상" disabled={busy} />
        {registering && <><label htmlFor="account-confirm">비밀번호 확인</label>
          <input id="account-confirm" name="confirm" type="password" autoComplete="new-password" required minLength={10} maxLength={128} disabled={busy} /></>}
        {error && <p className="account-error" role="alert">{error}</p>}
        <button className="account-primary" type="submit" disabled={busy}>{busy ? '처리 중...' : registering ? '가입하고 로그인' : '로그인'}</button>
      </form>
    </dialog>
  )
}

export default function AccountPanel({ selected, onSelect }) {
  const [user, setUser] = useState(null)
  const [items, setItems] = useState([])
  const [available, setAvailable] = useState(false)
  const [firebaseConfig, setFirebaseConfig] = useState(null)
  const [loading, setLoading] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const revision = useRef(0)
  const refresh = useCallback(async () => {
    const current = ++revision.current
    try {
      const session = await accountRequest('/auth/me')
      if (current !== revision.current) return
      setAvailable(session.available)
      setFirebaseConfig(session.provider === 'firebase' ? session.firebase_config : null)
      setUser(session.user)
      setItems([])
      setError(session.available ? '' : session.message)
      if (session.user) {
        const watchlist = await accountRequest('/watchlist')
        if (current === revision.current) setItems(watchlist.items)
      }
    } catch (failure) {
      if (current !== revision.current) return
      if (failure.status === 401) { setUser(null); setItems([]) }
      setError(failure.message)
    } finally { if (current === revision.current) setLoading(false) }
  }, [])
  useEffect(() => { refresh(); return () => { revision.current++ } }, [refresh])
  const saved = !!selected && items.some((item) => item.ticker === selected.ticker)
  async function mutate(action) {
    if (busy) return
    setBusy(true)
    setError('')
    revision.current++
    try { await action() }
    catch (failure) {
      if (failure.status === 401) { setUser(null); setItems([]) }
      setError(failure.message)
    } finally { setBusy(false) }
  }
  function toggle(stock, remove) {
    if (!user) { setDialogOpen(true); return }
    mutate(async () => {
      await accountRequest(`/watchlist/${encodeURIComponent(stock.ticker)}`, remove ? 'DELETE' : 'PUT')
      const watchlist = await accountRequest('/watchlist')
      setItems(watchlist.items)
    })
  }
  return (
    <section className="account-section" aria-label="내 계정과 관심종목">
      <div className="account-toolbar">
        <span className="account-identity"><UserRound size={18} /><strong>{loading ? '계정 확인 중...' : user?.username || '내 계정'}</strong></span>
        {user ? <button className="account-icon" type="button" title="로그아웃" aria-label="로그아웃" disabled={busy || loading}
          onClick={() => mutate(async () => { await accountRequest('/auth/logout', 'POST'); setUser(null); setItems([]) })}><LogOut size={18} /></button>
          : <button className="account-login" type="button" disabled={loading || !available} onClick={() => setDialogOpen(true)}><LogIn size={16} />로그인</button>}
      </div>
      <div className="section-head"><strong>관심종목</strong><span className="muted">{user ? `${items.length}개` : ''}</span></div>
      {selected && <button className={`watch-toggle ${saved ? 'is-saved' : ''}`} type="button" disabled={busy || loading || !available}
        aria-pressed={saved} onClick={() => toggle(selected, saved)}>
        <Star size={18} fill={saved ? 'currentColor' : 'none'} /><span>{selected.name || selected.stock_name || selected.ticker}</span><small>{saved ? '저장됨' : '추가'}</small>
      </button>}
      {!loading && !user && available && <p className="account-empty">로그인 후 이용할 수 있습니다.</p>}
      {!loading && user && items.length === 0 && !error && <p className="account-empty">저장된 관심종목이 없습니다.</p>}
      <div className="watch-list" aria-busy={busy}>
        {items.map((item) => <div className="watch-row" key={item.ticker}>
          <button type="button" className="watch-select" disabled={item.unavailable} onClick={() => onSelect(item)}>
            <strong>{item.name}</strong><small>{item.code}{item.market ? ` · ${item.market}` : ''}{item.unavailable ? ' · 조회 불가' : ''}</small>
          </button>
          <button type="button" className="account-icon" disabled={busy || loading} title="관심종목 삭제" aria-label={`${item.name} 관심종목 삭제`} onClick={() => toggle(item, true)}><Trash2 size={16} /></button>
        </div>)}
      </div>
      {error && <p className="account-error" role="alert">{error}</p>}
      {!loading && error && !busy && <button className="account-retry" type="button" onClick={() => { setLoading(true); refresh() }}>다시 시도</button>}
      {dialogOpen && <LoginDialog firebaseConfig={firebaseConfig} onClose={() => setDialogOpen(false)} onAuthenticated={(next) => { setUser(next); setLoading(true); refresh() }} />}
    </section>
  )
}
