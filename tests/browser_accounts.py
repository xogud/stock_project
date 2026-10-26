"""UI smoke test with mocked account APIs; never creates real Firebase users."""

import json
import os
from pathlib import Path

from playwright.sync_api import sync_playwright, expect

BASE = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8011")
OUT = Path(".cache/account-ui")
OUT.mkdir(parents=True, exist_ok=True)


def run():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        for width, height in [(1440, 1000), (390, 844)]:
            page = browser.new_page(viewport={"width": width, "height": height})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            state = {"user": None, "items": []}

            def account_route(route):
                path = route.request.url.split("/api", 1)[1]
                if path == "/auth/me":
                    result = {"user": state["user"], "available": True, "provider": "firebase", "firebase_config": {"projectId": "test-project"}}
                elif path == "/auth/logout":
                    state["user"] = None
                    result = {"ok": True}
                elif path.startswith("/watchlist/"):
                    state["items"] = []
                    result = {"ok": True}
                else:
                    result = {"items": state["items"]}
                route.fulfill(status=200, content_type="application/json", body=json.dumps(result))

            page.route("**/api/auth/**", account_route)
            page.route("**/api/watchlist**", account_route)
            page.goto(BASE, wait_until="domcontentloaded")
            page.get_by_role("button", name="로그인", exact=True).click()
            expect(page.get_by_role("dialog")).to_be_visible()
            expect(page.get_by_label("이메일", exact=True)).to_have_attribute("type", "email")
            page.get_by_role("tab", name="회원가입").click()
            page.get_by_label("이메일", exact=True).fill("test@example.com")
            page.get_by_label("비밀번호", exact=True).fill("example-password")
            page.get_by_label("비밀번호 확인", exact=True).fill("different-password")
            page.get_by_role("button", name="가입하고 로그인").click()
            expect(page.get_by_role("alert")).to_have_text("비밀번호 확인이 일치하지 않습니다.")
            box = page.get_by_role("dialog").bounding_box()
            assert box["x"] >= 0 and box["x"] + box["width"] <= width
            page.screenshot(path=str(OUT / f"register-{width}.png"))
            page.get_by_role("button", name="닫기", exact=True).click()
            state["user"] = {"id": "alice", "username": "long-email-for-layout-check@example.com"}
            state["items"] = [{"ticker": "005930.KS", "name": "삼성전자", "code": "005930", "market": "KOSPI"}]
            page.reload(wait_until="domcontentloaded")
            expect(page.get_by_role("button", name="삼성전자 관심종목 삭제")).to_be_visible()
            page.screenshot(path=str(OUT / f"watchlist-{width}.png"), full_page=True)
            page.get_by_role("button", name="삼성전자 관심종목 삭제").click()
            expect(page.get_by_text("저장된 관심종목이 없습니다.")).to_be_visible()
            page.get_by_role("button", name="로그아웃", exact=True).click()
            expect(page.get_by_role("button", name="로그인", exact=True)).to_be_enabled()
            assert not errors, errors
            page.close()
        browser.close()
    print("Desktop/mobile account UI checks passed (mocked Firebase APIs)")


if __name__ == "__main__":
    run()
