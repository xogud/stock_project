# Render 배포 방법

이 프로젝트는 Docker 기반 Render Web Service로 배포합니다.

## 1. GitHub에 프로젝트 올리기

현재 폴더는 Git 저장소가 아니므로 먼저 GitHub 저장소를 만들어야 합니다.

```powershell
git init
git add .
git commit -m "Prepare Render deployment"
git branch -M main
git remote add origin https://github.com/본인계정/stock-risk-monitor.git
git push -u origin main
```

## 2. Render에서 서비스 만들기

1. https://render.com 접속
2. New + 선택
3. Blueprint 선택
4. GitHub 저장소 연결
5. `render.yaml` 감지 확인
6. Apply 클릭

또는 Web Service로 직접 만들 경우:

- Runtime: Docker
- Dockerfile Path: `Dockerfile`
- Health Check Path: `/api/health`

## 3. 환경변수 설정

Render 서비스의 Environment 메뉴에서 설정합니다.

```text
MYSQL_ENABLED=0
DART_API_KEY=본인_DART_인증키
```

MySQL 없이 CSV와 모델 파일 기반으로 동작합니다.

## 4. 배포 후 접속

배포가 끝나면 Render가 아래와 같은 주소를 제공합니다.

```text
https://stock-risk-monitor.onrender.com
```

이 주소를 학교 밖 사용자에게 공유하면 됩니다.

## 5. 주의사항

- 무료 플랜은 처음 접속할 때 느릴 수 있습니다.
- 데이터가 최신으로 바뀌면 로컬에서 업데이트 후 다시 GitHub에 push해야 Render가 재배포합니다.
- DART 인증키는 코드에 직접 넣지 말고 Render 환경변수로만 설정합니다.
