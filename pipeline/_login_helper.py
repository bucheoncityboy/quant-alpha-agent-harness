#!/usr/bin/env python3
"""_login_helper.py — WQ 세션 + 생체인증 처리

SingleSession(ace_lib) 기반, 쿠키를 ~/secrets/.brain_cookies 에 저장/복원.
"""
import json, os, sys, time, logging
from pathlib import Path
from urllib.parse import urljoin
import requests

logger = logging.getLogger("wqlogin")

_SECRETS_DIR = Path(os.path.expanduser("~/secrets"))
COOKIE_FILE = _SECRETS_DIR / ".brain_cookies"
PERSONA_FILE = _SECRETS_DIR / ".brain_persona"
BRAIN_API = os.environ.get("BRAIN_API_URL", "https://api.worldquantbrain.com")


def _creds():
    email = os.environ.get("BRAIN_ID")
    pw = os.environ.get("BRAIN_PASSWORD")
    if email and pw:
        return (email, pw)
    secrets = Path(os.path.expanduser("~/secrets/platform-brain.json"))
    if secrets.exists() and secrets.stat().st_size > 2:
        data = json.loads(secrets.read_text(encoding="utf-8"))
        return (data["email"], data["password"])
    print(json.dumps({"error": "No credentials. Fill BRAIN_ID/BRAIN_PASSWORD in .env"}, indent=2))
    sys.exit(1)


def _save_cookies(s):
    cookies = [{"n": c.name, "v": c.value, "d": c.domain, "p": c.path} for c in s.cookies]
    _SECRETS_DIR.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(json.dumps(cookies, ensure_ascii=False), encoding="utf-8")


def _load_cookies(s):
    if COOKIE_FILE.exists():
        for c in json.loads(COOKIE_FILE.read_text(encoding="utf-8")):
            s.cookies.set(c["n"], c["v"], domain=c.get("d", ""), path=c.get("p", "/"))
        return True
    return False


def get_session():
    from engine.ace_lib import SingleSession

    s = SingleSession()
    s.auth = _creds()
    _load_cookies(s)

    # 0) 쿠키로 세션 유효하면 바로 리턴
    r = s.get(f"{BRAIN_API}/authentication")
    if r.status_code == 200:
        _save_cookies(s)
        return s

    # 1) PERSONA_FILE 있으면 → 마무리 POST
    if PERSONA_FILE.exists():
        persona = json.loads(PERSONA_FILE.read_text(encoding="utf-8"))
        loc = persona.get("location_url")
        if loc:
            print(json.dumps({"info": "completing biometric auth..."}, indent=2))
            s.post(loc)
        PERSONA_FILE.unlink(missing_ok=True)
        r = s.get(f"{BRAIN_API}/authentication")
        if r.status_code == 200:
            _save_cookies(s)
            return s

    # 2) POST /authentication 로그인 시도
    r = s.post(f"{BRAIN_API}/authentication")
    if r.status_code == 401 and r.headers.get("WWW-Authenticate") == "persona":
        loc = urljoin(r.url, r.headers["Location"])
        print(json.dumps({
            "error": "biometric_auth_required",
            "url": loc,
            "instruction": "브라우저에서 URL 열고 인증 완료 후, 다시 실행하세요."
        }, indent=2, ensure_ascii=False))
        PERSONA_FILE.write_text(json.dumps({
            "location_url": loc, "timestamp": time.time()
        }), encoding="utf-8")
        sys.exit(1)

    if r.status_code == 401:
        print(json.dumps({"error": "Incorrect email or password"}, indent=2))
        sys.exit(1)

    _save_cookies(s)
    return s
