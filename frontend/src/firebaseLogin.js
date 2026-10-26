import { getApps, initializeApp } from 'firebase/app'
import {
  createUserWithEmailAndPassword, getAuth, inMemoryPersistence,
  setPersistence, signInWithEmailAndPassword, signOut,
} from 'firebase/auth'

const messages = {
  'auth/invalid-credential': '이메일 또는 비밀번호가 올바르지 않습니다.',
  'auth/invalid-email': '이메일 주소를 확인해 주세요.',
  'auth/email-already-in-use': '이미 가입된 이메일입니다. 로그인해 주세요.',
  'auth/weak-password': '더 강한 비밀번호를 사용해 주세요.',
  'auth/too-many-requests': '시도가 너무 많습니다. 잠시 후 다시 시도해 주세요.',
  'auth/network-request-failed': '네트워크 연결을 확인해 주세요.',
  'auth/operation-not-allowed': '이메일 로그인이 아직 활성화되지 않았습니다.',
}

export async function firebaseLogin(config, registering, email, password, exchange) {
  const app = getApps().find((item) => item.name === 'stock-risk') || initializeApp(config, 'stock-risk')
  const auth = getAuth(app)
  try {
    // Only the HttpOnly backend session persists, not browser access tokens.
    await setPersistence(auth, inMemoryPersistence)
    const login = registering ? createUserWithEmailAndPassword : signInWithEmailAndPassword
    const credential = await login(auth, email.trim(), password)
    return await exchange(await credential.user.getIdToken())
  } catch (error) {
    if (error.code?.startsWith('auth/')) {
      throw new Error(messages[error.code] || '로그인하지 못했습니다. 계정 설정을 확인해 주세요.')
    }
    throw error
  } finally {
    await signOut(auth)
  }
}
