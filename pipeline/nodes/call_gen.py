"""call_gen node — converts raw calls to BehaviorEventEntity, with LLM transcript generation.

Transcript generation:
  - Batch: up to BATCH_SIZE calls per LLM call to reduce API round trips
  - Missed calls: no transcript (empty string)
  - test_mode: skips LLM, leaves transcriptDialog empty
  - Format: realistic Korean dialogue
      A: 여보세요?
      B: 안녕하세요, ...
      ...
"""
import json
from datetime import datetime

from langchain_core.messages import HumanMessage

from pipeline.full_state import FullPipelineState

BATCH_SIZE = 10  # calls per LLM request


# ── Helpers ───────────────────────────────────────────────────────────────────

def _datetime_to_unix_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def _get_llm(provider: str):
    p = (provider or "claude").lower()
    if p == "claude":
        from langchain_anthropic import ChatAnthropic
        from llm.config import CLAUDE_MODEL, ANTHROPIC_API_KEY
        return ChatAnthropic(model=CLAUDE_MODEL, api_key=ANTHROPIC_API_KEY, max_tokens=4096)
    if p == "gpt":
        from langchain_openai import ChatOpenAI
        from llm.config import GPT_MODEL, OPENAI_API_KEY
        return ChatOpenAI(model=GPT_MODEL, api_key=OPENAI_API_KEY, max_tokens=4096)
    if p == "glm":
        from langchain_openai import ChatOpenAI
        from llm.config import GLM_MODEL, GLM_API_KEY, GLM_BASE_URL
        return ChatOpenAI(model=GLM_MODEL, api_key=GLM_API_KEY, base_url=GLM_BASE_URL, max_tokens=4096)
    raise ValueError(f"Unknown provider: {p!r}")


def _convert_call(call: dict) -> dict:
    """Convert raw call dict to BehaviorEventEntity (transcriptDialog filled later)."""
    timestamp = _datetime_to_unix_ms(call["datetime"])

    if call.get("datetime_end"):
        ts_end = _datetime_to_unix_ms(call["datetime_end"])
        duration = (ts_end - timestamp) // 1000
    else:
        duration = 0

    call_result = call.get("call_result", "")
    direction = call.get("direction", 0)

    if "missed" in call_result.lower():
        call_type = 2   # Missed
    elif direction == 0:
        call_type = 0   # Outgoing
    else:
        call_type = 1   # Incoming

    payload = {
        "number": call.get("phoneNumber", ""),
        "name": call.get("contactName", ""),
        "date": timestamp,
        "type": call_type,
        "geocodedLocation": "",
        "duration": duration,
        "missedReason": "" if call_type != 2 else "No answer",
        "transcriptDialog": "",   # filled by _generate_transcripts_batch
    }

    return {
        "identifier": f"call_{call['event_id']}_{timestamp}",
        "timestamp": timestamp,
        "event_source": "CALL_LOG",
        "payload": json.dumps(payload, ensure_ascii=False),
        "contextGroupId": None,
        "_raw": call,   # temporary: removed before final output
    }


def _find_relationship(persona: dict, contact_name: str) -> str:
    """Return relationship label for contact_name from persona.relationships."""
    for r in persona.get("relationships", []):
        name = r.get("name") or r.get("이름") or ""
        if name == contact_name:
            return r.get("relation") or r.get("관계") or ""
    return ""


def _generate_transcripts_batch(
    llm,
    persona: dict,
    calls: list[dict],     # raw call dicts (not BehaviorEventEntity)
) -> dict[int, str]:
    """Generate transcripts for a batch of calls. Returns {event_id: transcript}."""

    persona_name = persona.get("name", "")
    persona_job  = persona.get("job", "")

    # Build batch description
    items = []
    for call in calls:
        contact = call.get("contactName", "")
        relation = _find_relationship(persona, contact)
        direction = "발신" if call.get("direction", 0) == 0 else "수신"
        duration_s = 0
        if call.get("datetime_end"):
            ts  = _datetime_to_unix_ms(call["datetime"])
            ts2 = _datetime_to_unix_ms(call["datetime_end"])
            duration_s = (ts2 - ts) // 1000
        date_str = call.get("datetime", "")[:10]
        eid = call["event_id"]
        rel_str = f"({relation})" if relation else ""
        items.append(
            f"[ID:{eid}] {date_str} | {direction} | 상대: {contact}{rel_str} | 통화시간: {duration_s}초"
        )

    batch_text = "\n".join(items)

    prompt = f"""아래 인물이 진행한 통화 목록에 대해 각각 현실적인 통화 대화 스크립트를 생성하세요.

## 인물 정보
- 이름: {persona_name}
- 직업: {persona_job}

## 통화 목록
{batch_text}

## 생성 요구사항
1. 각 통화 ID([ID:N])별로 대화 스크립트 작성
2. 실제 한국어 구어체 대화 형식: "A: 대사\\nB: 대사\\nA: 대사..." (3~8 턴)
3. 통화 방향(발신/수신), 통화 시간, 관계를 반영
4. 짧은 통화(60초 이하)는 간단하게, 긴 통화는 내용 있게
5. 인물의 직업과 상대방 관계에 자연스러운 내용

## 출력 형식 (JSON)
각 ID를 키로 하는 JSON 객체만 출력:
{{
  "ID1": "A: 여보세요?\\nB: 안녕하세요...",
  "ID2": "A: ...",
  ...
}}
JSON만 출력하세요."""

    try:
        from json_repair import repair_json
        import re
        resp = llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(repair_json(raw.strip()))
        # Normalize keys: "ID123" or "123" → int
        result = {}
        for k, v in parsed.items():
            k_clean = k.replace("ID", "").strip()
            try:
                result[int(k_clean)] = str(v)
            except ValueError:
                pass
        return result
    except Exception as e:
        print(f"[call_gen] transcript batch error: {e}")
        return {}


# ── Node ──────────────────────────────────────────────────────────────────────

def generate_calls(state: FullPipelineState) -> FullPipelineState:
    """Convert raw_calls → generated_calls with LLM transcript generation."""
    provider  = state.get("provider", "claude")
    test_mode = state.get("test_mode", False)
    personas  = state.get("personas", [])

    # Build persona lookup by name
    persona_by_name: dict = {p.get("name", ""): p for p in personas}

    # First pass: convert all calls
    converted: list[dict] = []
    for call in state.get("raw_calls", []):
        try:
            converted.append(_convert_call(call))
        except Exception as e:
            print(f"[call_gen] skipping {call.get('event_id')}: {e}")

    if test_mode:
        # Remove _raw and return as-is (no transcript)
        results = []
        for ev in converted:
            ev.pop("_raw", None)
            results.append(ev)
        return {**state, "generated_calls": results}

    # Second pass: LLM transcript generation in batches per persona
    llm = _get_llm(provider)

    # Group non-missed calls by persona
    persona_calls: dict[str, list] = {}
    for ev in converted:
        raw = ev.get("_raw", {})
        pname = raw.get("persona_name", "")
        payload = json.loads(ev["payload"])
        if payload["type"] == 2:  # missed → no transcript needed
            continue
        persona_calls.setdefault(pname, []).append(raw)

    # Generate transcripts
    transcript_map: dict[int, str] = {}  # event_id → transcript
    for pname, calls_for_persona in persona_calls.items():
        persona = persona_by_name.get(pname, {})
        print(f"[call_gen] {pname}: generating transcripts for {len(calls_for_persona)} calls...")
        for i in range(0, len(calls_for_persona), BATCH_SIZE):
            batch = calls_for_persona[i: i + BATCH_SIZE]
            batch_result = _generate_transcripts_batch(llm, persona, batch)
            transcript_map.update(batch_result)
            print(f"[call_gen] {pname}: batch {i//BATCH_SIZE + 1} → {len(batch_result)} transcripts")

    # Third pass: inject transcripts into payloads
    results = []
    for ev in converted:
        raw = ev.pop("_raw", {})
        payload = json.loads(ev["payload"])
        eid = raw.get("event_id")
        if eid is not None and payload["type"] != 2:
            transcript = transcript_map.get(int(eid), "")
            payload["transcriptDialog"] = transcript
            ev["payload"] = json.dumps(payload, ensure_ascii=False)
        results.append(ev)

    print(f"[call_gen] Total {len(results)} calls, {len(transcript_map)} transcripts generated")
    return {**state, "generated_calls": results}
