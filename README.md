# quant-alpha-agent-harness — WorldQuant BRAIN 알파 마이닝 에이전트 하네스

> LLM 에이전트가 가설 생성부터 시뮬레이션·통계 검증·제출까지 전 과정을 통제하는 WorldQuant BRAIN 알파 마이닝 하네스. 무작위 조합이 아니라 에이전트가 매 단계 판단한다.

## ⭐ 성과

- **유효 알파 70개 발굴**, 플랫폼 기준 통과 시그널을 지속 확보 (제출 기록 14건)
- **통계 가드레일 + 5-Gate 제출 파이프라인**, 자격 미달 알파는 제출 경로가 차단되는 Fail-Closed 구조
- **리서치 컨설턴트 계약 체결**, WorldQuant 산하 컨설턴트와 실전 운영
- **NDA-safe 공개**: 개별 알파 수식 원문과 데이터셋 카탈로그는 플랫폼 규정에 따라 비공개. 에이전트 제어 프레임워크만 공개한다.

## 검증 대상 질문

"LLM 에이전트가 실제로 유효한 알파 시그널을 발굴할 수 있는가?"

무작위 조합(auto-loop)이 아니라 에이전트가 조합·검토·제출 결정을 내리는 대화형 마이닝 워크플로우를 구축했다. 검증 대상은 두 가지다.

1. 에이전트의 표현식 생성이 통계적으로 의미 있는가 (단순 조합 대비)
2. 에이전트의 제출 판단이 가드레일을 통과하는가 (Fail-Closed)

## 시스템 설계

```
LLM Agent
  연산자/필드/구성 선택 · 시뮬레이션 결과 검토 · 다음 단계 결정
                     ▼
alpha_pipeline_v3.py — CLI 5-Gate
  --validate  66개 연산자 + 7,958개 필드 토큰 검증
  --compose   검증 + 설정 자동 해석 (universe/decay/중립화)
  --simulate  단일 시뮬레이션 (rate-limiter 통제)
  --corr      기존 제출 풀 대비 자기상관 검사
  --check     제출 자격 5-Gate
  --submit    ACTIVE 상태만 제출
                     ▼
Validation Gate 계층
  · 연산자 화이트리스트 (66개)
  · 필드 레지스트리 검증 (플랫폼 CSV 로컬 인덱스)
  · 3-Level Dedup + 자동 Pivot 변형 제안
  · 상태 머신 영속화 (JSON/SQLite)
  · Rate Limiter (3병렬, 429 지수 백오프 60→900s)
                     ▼
         WorldQuant BRAIN API
```

**설계 원칙**

- **No auto-loop**: 무작위 조합 생성이 아니라 에이전트가 매 스텝 판단
- **Fail-Closed**: 조건 미충족 시 제출 경로 차단 (중복/토큰/설정 검증 실패 = 거부)
- **상태 영속화**: 시뮬레이션·제출 이력이 로컬에 남아 중복·상관 참조의 기준이 됨

## 핵심 컴포넌트

| 모듈 | 역할 |
|---|---|
| `alpha_researcher.py` | 가설 검증·조합 로직 (에이전트 통제) |
| `pivot_engine.py` | 중복 회피 Pivot 변형 생성 |
| `my_alpha_pool.py` | 제출 풀 관리·3단계 중복 탐지 (정확/구조/시그니처) |
| `state_manager.py` | 세션·결정 이력 영속화 |
| `sim_runner.py` | 시뮬레이션 실행·결과 파싱 |
| `field_registry.py` | 7,958개 필드 인덱스 |
| `rate_limiter.py` | 3병렬 + 429 지수 백오프 |

## 검증

- **가드레일 실증**: pytest 4개 스위트(알파 리서처 조합 로직, 필드 레지스트리 로딩, Pivot 엔진, 상태 관리자)
- **중복 방지**: 제출 풀 대비 정확/구조적/시그니처 3단계 탐지, 구조적 중복 시 자동 Pivot 변형 제안
- **제출 파이프라인**: 5-Gate 통과 후에만 `POST /alphas/{id}/submit`, ACTIVE 상태만 풀에 반영

## 저장소 구조

```
pipeline/
  alpha_pipeline_v3.py     # CLI 오케스트레이터 (5-Gate)
  simulate.py / submit.py  # 단일 알파 시뮬레이션·제출 래퍼
  list_active.py           # ACTIVE 알파 조회
  download_fields.py       # 플랫폼 필드 카탈로그 로컬 동기화
  engine/                  # 에이전트·중복 탐지·상태 관리·레이트리밋
  tests/                   # pytest 4개 스위트
```

## License

프레임워크: MIT. 플랫폼 데이터·알파 수식은 WorldQuant BRAIN 이용약관 적용.
