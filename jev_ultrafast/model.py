"""TypeSafe makes choices; an optional small OpenAI-compatible model writes field values."""

import json
import math
import os
import time

import httpx

from .questions import NEXT_ACTION, TARGET, TEXT_VALUE

CLIENT = httpx.Client(http2=True, timeout=25)


def post_json(url, key, body):
    for attempt in range(3):
        try:
            response = CLIENT.post(url, json=body, headers={"Authorization": f"Bearer {key}"})
        except httpx.HTTPError:
            raise RuntimeError("Model connection failed; no action executed.") from None
        if response.status_code in {429, 529, 503} and attempt < 2:
            time.sleep(0.5 * 2**attempt)
            continue
        if response.is_error:
            detail = response.text[:300].replace(key, "<key>") if key else response.text[:300]
            raise RuntimeError(f"Model provider returned HTTP {response.status_code}: {detail}; no action executed.")
        return response.json()
    raise RuntimeError("Model unavailable")


def validate_choice(answer, ids):
    try:
        probabilities = answer["probabilities"]
        numbers = [*probabilities.values(), answer["confidence"]]
        valid = (
            answer["choice"] in ids
            and set(probabilities) == set(ids)
            and all(type(n) in (int, float) and math.isfinite(n) and 0 <= n <= 1 for n in numbers)
            and abs(sum(probabilities.values()) - 1) < 0.02
            and probabilities[answer["choice"]] >= max(probabilities.values()) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Invalid TypeSafe response; no action executed.")
    return answer


def action_space(actions):
    """One index per observed element; each operation has its own valid target choices."""
    elements, indices, targets, controls = [], {}, {}, {}
    operations = {"click": "CLICK", "fill": "TYPE_TEXT", "select": "SELECT"}
    for action in actions:
        kind = action["kind"]
        if kind not in operations:
            controls[action["id"].upper()] = action
            continue
        node = action["node"]
        if node not in indices:
            index = str(len(elements) + 1)
            indices[node] = index
            element = {k: action[k] for k in ("role", "value", "checked", "selected", "expanded") if k in action}
            element.update(index=index, label=action["label"].split(" → ")[0], operations=[])
            if kind == "select":
                element["value"] = action.get("current_value", "")
                element["options"] = []
            elements.append(element)
        index = indices[node]
        operation = operations[kind]
        group = targets.setdefault(operation, {})
        element = elements[int(index) - 1]
        if operation not in element["operations"]:
            element["operations"].append(operation)
        target = index
        if kind == "select":
            target = f"{index}:{len(element['options']) + 1}"
            element["options"].append({"index": target, "label": action["label"], "value": action["value"]})
        group[target] = action
    return elements, targets, controls


# TypeSafe Jev accepts up to 255 options per choice question; Ollama's Clef accepts 26.
MAX_CANDIDATES = int(os.environ.get("CHOICE_MAX_CANDIDATES", "255"))


def candidate_groups(operation, candidates):
    """Question name -> candidates, each group within 2..MAX_CANDIDATES. A lone candidate gets no question."""
    items = list(candidates.items())
    if len(items) < 2:
        return {}
    name = operation.lower() + "_target"
    count = math.ceil(len(items) / MAX_CANDIDATES)
    size = math.ceil(len(items) / count)  # even split, so no group is left with a single candidate
    chunks = [dict(items[i : i + size]) for i in range(0, len(items), size)]
    if len(chunks) == 1:
        return {name: chunks[0]}
    return {f"{name}_{n}": chunk for n, chunk in enumerate(chunks, 1)}


def target_question(operation, candidates, goal):
    return {
        "type": "choice",
        "criteria": {
            index: {
                "element": f"[{index}] {a['label']}",
                "current_value": a.get("current_value", a.get("value", "")),
                **{k: a[k] for k in ("role", "checked", "selected", "expanded") if k in a},
            }
            for index, a in candidates.items()
        },
        "instructions": {"goal": goal, "operation": operation, "rules": [NEXT_ACTION, TARGET]},
    }


def choose(state, goal, history):
    elements, targets, controls = action_space(state["actions"])
    labels = {
        "CLICK": "Click an element, button, menu option, autocomplete suggestion, or calendar day.",
        "TYPE_TEXT": "Enter or replace text in an editable field. A small LLM will supply the value from the goal.",
        "SELECT": "Select an observed dropdown value.",
    }
    operations = {key: labels[key] for key in targets}
    operations.update({key: value["label"] for key, value in controls.items()})
    operations.update(DONE="Every requirement is visibly satisfied.", BLOCKED="No supported operation can progress.")
    questions = {
        "operation": {"type": "choice", "criteria": operations, "instructions": {"goal": goal, "rules": NEXT_ACTION}}
    }
    groups = {}
    for operation, candidates in targets.items():
        groups[operation] = candidate_groups(operation, candidates)
        for name, group in groups[operation].items():
            questions[name] = target_question(operation, group, goal)
    body = {
        "model": os.environ.get("TYPESAFE_MODEL", "jev-latest"),
        "state": {
            "page": {k: state[k] for k in ("url", "title", "text")},
            "elements": elements,
            "recent_actions": [
                {k: h.get(k) for k in ("action", "kind", "text", "page_changed")} for h in history[-10:]
            ],
        },
        "questions": questions,
    }
    started = time.perf_counter()
    # Any server speaking the System One API: TypeSafe (default), LiteLLM, or Ollama serving Clef.
    base = os.environ.get("TYPESAFE_API_BASE", "https://api.typesafe.ai").rstrip("/")
    result = post_json(base + "/v1/systemone", os.environ["TYPESAFE_API_KEY"], body)
    operation_answer = validate_choice(result["answers"].get("operation", {}), operations)
    operation = operation_answer["choice"]
    target = None
    target_answer = None
    probabilities = {}
    if operation in targets:
        # Unused target heads cannot cause an action. Validate the head selected by the operation.
        candidates, group_questions = targets[operation], groups[operation]
        if not group_questions:  # a single candidate needs no question
            only = next(iter(candidates))
            target_answer = {"choice": only, "probabilities": {only: 1.0}, "confidence": 1.0}
        elif len(group_questions) == 1:
            ((name, group),) = group_questions.items()
            target_answer = validate_choice(result["answers"].get(name, {}), group)
        else:
            # Candidates were split to fit the server's per-question limit: ask again among group winners.
            winners = {}
            for name, group in group_questions.items():
                winner = validate_choice(result["answers"].get(name, {}), group)["choice"]
                winners[winner] = candidates[winner]
            name = operation.lower() + "_target"
            final = {**body, "questions": {name: target_question(operation, winners, goal)}}
            final_result = post_json(base + "/v1/systemone", os.environ["TYPESAFE_API_KEY"], final)
            target_answer = validate_choice(final_result["answers"].get(name, {}), winners)
        target = target_answer["choice"]
        choice = candidates[target]["id"]
        probabilities = {a["id"]: target_answer["probabilities"].get(index, 0.0) for index, a in candidates.items()}
    else:
        choice = controls[operation]["id"] if operation in controls else operation
        probabilities[choice] = operation_answer["probabilities"][operation]
    return {
        "choice": choice,
        "operation": operation,
        "target": target,
        "confidence": operation_answer["confidence"],
        "probabilities": probabilities,
        "operation_probabilities": operation_answer["probabilities"],
        "target_probabilities": target_answer["probabilities"] if target_answer else {},
        "target_confidence": target_answer["confidence"] if target_answer else None,
        "raw_answers": result["answers"],
        "model": result["model"],
        "usage": result.get("usage", {}),
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "request": body,
    }


def field_context(goal, action, page, history):
    return {
        "goal": goal,
        "field": {k: action.get(k) for k in ("label", "role", "value")},
        "page": {"title": page["title"], "text": page["text"][:6000]},
        "recent_actions": [{k: h.get(k) for k in ("action", "text")} for h in history[-6:]],
    }


def field_text(context):
    key = os.environ.get("TEXT_MODEL_API_KEY")
    if not key:
        raise ValueError("TYPE_TEXT needs TEXT_MODEL_API_KEY; no text is hardcoded or guessed by the executor.")
    base = os.environ.get("TEXT_MODEL_BASE_URL", "https://api.deepseek.com/v1").rstrip("/")
    model = os.environ.get("TEXT_MODEL", "deepseek-chat")
    reasoning = {"thinking": {"type": "disabled"}} if "api.deepseek.com/" in base else {"reasoning": {"effort": "low"}}
    if os.environ.get("TEXT_MODEL_REASONING") == "none":
        reasoning = {"reasoning": {"enabled": False}}
    # Servers differ on how to turn thinking off (Ollama: {"reasoning_effort": "none"}); this overrides the above.
    if extra := os.environ.get("TEXT_MODEL_EXTRA_BODY"):
        reasoning = json.loads(extra)
    started = time.perf_counter()
    result = post_json(
        base + "/chat/completions",
        key,
        {
            "model": model,
            "max_tokens": 1024,
            "response_format": {"type": "json_object"},
            **reasoning,
            "messages": [
                {"role": "system", "content": TEXT_VALUE},
                {
                    "role": "user",
                    "content": json.dumps(context),
                },
            ],
        },
    )
    try:
        output = json.loads(result["choices"][0]["message"]["content"])
        value = output["text"]
        if set(output) != {"text"} or not isinstance(value, str) or not value.strip() or len(value) > 2000:
            raise ValueError()
    except (ValueError, KeyError, TypeError):
        raise ValueError("Text helper returned no valid field value; nothing typed.") from None
    return value, {
        "model": model,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "usage": result.get("usage", {}),
    }
