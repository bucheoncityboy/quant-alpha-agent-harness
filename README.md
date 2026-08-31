# quant-alpha-agent-harness

> **One-Line Pitch:** LLM 에이전트가 가설 생성 → 시뮬레이션 → 통계 검증 → 제출까지 전 과정을 통제하는 WorldQuant BRAIN 알파 마이닝 하네스

## 📌 Executive Summary

| Category | Details |
| :--- | :--- |
| **Core Objective** | "Agent를 활용해 실제 알파 시그널을 발굴할 수 있는가" 검증. 무작위 조합(auto-loop)이 아니라 **LLM 에이전트가 모든 조합·검토·제출 결정을 내리는** 대화형 마이닝 워크플로우 구축 |
| **Key Architecture** | LLM Agent ↔ CLI 5-Gate (`validate → compose → simulate → corr → check → submit`) ↔ 통계적 가드레일 ↔ WorldQuant BRAIN API. 상태 머신(State Manager)으로 세션·결정 이력 영속화 |
| **Performance** | **66개 연산자 허용목록 검증** · **7,958개 필드 레지스트리** · **3단계 중복 탐지**(정확/구조/시그니처)+자동 Pivot 제안 · 플랫폼 기록 기준 **제출 알파 14건** (목표 1,000+) · Rate Limiter(3병렬, 429 백오프 60→900s) |
| **Tech Stack** | Python, WorldQuant BRAIN REST API, SQLite(상태·풀 캐시), pytest, Claude Code/Codex/가재 에이전트 |

> ⚠️ **NDA 준수**: WorldQuant 플랫폼 규정에 따라 개별 알파 수식 원문·데이터셋 카탈로그는 공개 범위에서 제외. 본 저장소는 **에이전트 제어 프레임워크**만 공개하며, 플랫폼 데이터(필드 카탈로그 CSV 등)는 로컬 환경에서 `download_fields.py`로 확보합니다.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  LLM Agent (GJC — human-like reasoning)                     │
│  연산자/필드/구성 선택 · 결과 검토 · 다음 단계 결정          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  alpha_pipeline_v3.py (CLI 5-Gate)                          │
│  --validate  토큰 검증 (66 연산자 + 7,958 필드)             │
│  --compose   검증 + 설정 자동 해석 (universe/decay/중립화)  │
│  --simulate  단일 시뮬레이션 (rate-limiter 통제)            │
│  --corr      자기상관 검사 (기존 제출 풀 대비)              │
│  --check     제출 자격 5-Gate                                │
│  --submit    ACTIVE 제출                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  Validation Gate 계층                                        │
│  · 허용 연산자 화이트리스트 (66)                             │
│  · 필드 레지스트리 검증 (플랫폼 CSV 로컬 인덱스)             │
│  · 3-Level Dedup + suggest_pivot (구조적 유사 대체안 생성)   │
│  · 상태 머신 영속화 (state_manager → JSON/SQLite)            │
│  · Rate Limiter (3 병렬, 429 지수 백오프)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
                  WorldQuant BRAIN API
```

**설계 원칙**
- **No auto-loop** — 무작위 조합 생성이 아니라 에이전트가 매 스텝 판단
- **Fail-Closed** — 조건 미충족 시 제출 경로 차단(중복/토큰/설정 검증 실패 = 거부)
- **상태 영속화** — 시뮬레이션·제출 이력이 로컬에 남아 중복·상관 참조의 기준이 됨

## 📊 Key Results & Validation

- **가드레일 실증**: 4개 테스트 스위트(`tests/`) — 알파 리서처 조합 로직, 필드 레지스트리 로딩, Pivot 엔진, 상태 관리자
- **중복 방지**: 제출 풀(`submitted_alphas.json`) 대비 정확/구조적/시그니처 3단계 탐지, 구조적 중복 시 자동 Pivot 변형 제안
- **제출 파이프라인**: 제출 자격 5-Gate 통과 후에만 `POST /alphas/{id}/submit` (ACTIVE 상태만 풀에 반영)

## 🛠️ 저장소 구조

```
pipeline/
  alpha_pipeline_v3.py     # CLI 오케스트레이터 (5-Gate)
  simulate.py / submit.py  # 단일 알파 시뮬레이션·제출 래퍼
  list_active.py           # ACTIVE 알파 조회
  download_fields.py       # 플랫폼 필드 카탈로그 로컬 동기화
  engine/
    alpha_researcher.py    # 가설 검증·조합 로직 (에이전트 통제)
    pivot_engine.py        # 중복 회피 Pivot 변형 생성
    my_alpha_pool.py       # 제출 풀 관리·3단계 중복 탐지
    state_manager.py       # 세션·결정 이력 영속화
    sim_runner.py          # 시뮬레이션 실행·결과 파싱
    field_registry.py      # 필드 레지스트리 인덱스
    settings_utils.py      # 설정 자동 해석 (delay/중립화)
    rate_limiter.py        # 3병렬 + 429 지수 백오프
  tests/                   # pytest (알파 리서처·레지스트리·피벗·상태)
```

## 🔬 실행 예시

```bash
# 1. 필드 카탈로그 로컬 준비 (플랫폼 세션 필요)
python pipeline/download_fields.py

# 2. 에이전트가 조합한 표현식 검증 → 시뮬레이션 → 제출
python alpha_pipeline_v3.py --validate "ts_zscore(close, 60)"
python alpha_pipeline_v3.py --simulate --alpha-id <id>
python alpha_pipeline_v3.py --corr --alpha-id <id>
python alpha_pipeline_v3.py --check --alpha-id <id>
python alpha_pipeline_v3.py --submit --alpha-id <id>
```

## License

프레임워크: MIT. 플랫폼 데이터·알파 수식은 WorldQuant BRAIN 이용약관 적용.