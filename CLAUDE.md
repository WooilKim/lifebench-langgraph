# CLAUDE.md — 프로젝트 핵심 목표

이 파일은 이 프로젝트의 **존재 이유**를 담는다.
개발하기 전 반드시 읽고, 항상 이 맥락 안에서 판단하라.

---

## 1. LifeBench는 무엇을 어떻게 만들었나

LifeBench는 **중국인 인물 10명**의 1년치 생활 데이터를 LLM으로 시뮬레이션한 벤치마크 데이터셋이다.

**데이터 생성 원칙:**
- 인물 프로파일(persona.json)을 기반으로 성격·직업·관계망이 반영된 생활 이벤트 생성
- 이벤트에서 스마트폰 행동 데이터(통화·문자·알림·캘린더 등) 파생
- 중국어 프롬프트로 구동 → 중국 앱(위챗·알리페이·디디 등) 기반 데이터
- 통화 내용(`transcriptDialog`)은 **항상 빈 문자열** — LLM이 생성하지 않음
- LLM 호출 방식: 이벤트 컨텍스트 → 프롬프트 → JSON 직접 출력 (파이프라인 없음)

**생성 데이터 구조:**
```
call.json     통화 기록 (방향·시간·상대방)
sms.json      문자 기록 (내용·상대방·방향)
push.json     앱 알림 (앱 이름·내용)
→ BehaviorEventEntity 포맷으로 변환 (lifebench_trans)
```

---

## 2. 우리가 만들려는 데이터와 어떻게 다른가

| 항목 | LifeBench 원본 | 우리가 필요한 것 |
|------|--------------|----------------|
| 언어 | 중국어 기반 | **한국어** 또는 다국어 |
| 앱 생태계 | 중국 앱 (위챗·알리페이 등) | **한국 앱** (카카오·네이버·토스 등) |
| 통화 내용 | 항상 빈 문자열 | **실제 대화 내용 요약** |
| LLM 의존성 | 고정 프롬프트 + 단일 중국어 LLM | **교체 가능** (GLM·Claude·GPT) |
| 파이프라인 | 스크립트 직접 실행 | **LangGraph 노드** — 교체·확장 가능 |
| 데이터 타겟 포맷 | BehaviorEventEntity (Android Room) | 유연하게 변경 가능 |
| 품질 제어 | 없음 | 노드별 검증·재시도 가능 |

---

## 3. LifeBench에 어떤 변화를 주었나

LangGraph를 사용해 각 생성 단계를 **교체 가능한 노드**로 분리했다.

**베이스 그래프 (원본 로직 그대로):**
```
loader → call_gen → sms_gen → noti_gen → formatter → END
```
- 규칙 기반, LLM 없음, LifeBench 원본 convert.py 로직 그대로

**변형 1: `korean_local`**
- `noti_gen` 노드 교체
- 앱 이름·알림 내용을 한국 앱 기반으로 로컬라이징
- 카카오톡·네이버·카카오페이 등 한국 앱 패키지명 매핑

**변형 2: `llm_enriched`**
- `call_gen` + `sms_gen` 노드 교체
- LLM으로 `transcriptDialog` (통화 내용 요약) 생성
- 이벤트 맥락을 프롬프트에 주입 → 자연스러운 대화 내용
- 환경변수로 GLM / Claude / GPT 전환 가능

**교체 방법:**
```python
# 노드 하나만 바꿔서 새 그래프 생성
graph = base_graph.copy()
graph.nodes["noti_gen"] = my_custom_noti_node
```

---

## 4. 앞으로 추가할 변형들

- `gmail_enriched`: 이메일 데이터 생성 (현재 LifeBench에 없음)
- `health_enriched`: 헬스 데이터를 이벤트 맥락 기반으로 보강
- `full_korean`: 모든 노드를 한국 생태계 기반으로 교체
- `custom_schema`: BehaviorEventEntity 외 다른 타겟 스키마로 출력

---

## 개발 원칙

1. **노드는 독립적으로 교체 가능해야 한다** — 다른 노드에 영향 없음
2. **LLM은 환경변수로 선택한다** — `LLM_PROVIDER=claude|gpt|glm`
3. **베이스 그래프는 수정하지 않는다** — variants/에 변형만 추가
4. **비교 결과는 항상 저장한다** — `viz/comparison_result.json`
5. **커밋은 단계별로** — 노드 하나 추가/교체할 때마다 커밋

---

## 참조 파일

- `~/source/LifeBench/` — 원본 LifeBench 코드
- `~/source/lifebench_trans/convert.py` — BehaviorEventEntity 변환 로직
- `~/source/lifebench_trans/format.txt` — 타겟 포맷 스키마
- `~/source/lifebench-viz/` — LifeBench 파이프라인 시각화 (참고용)
