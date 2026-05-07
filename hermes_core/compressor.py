"""Standalone 3-phase context compression inspired by Hermes agent.

Phases: prune -> protect -> summarize.
No Hermes internals. Injectable LLM function. Simple tokenizer.
"""

from __future__ import annotations

import random
from typing import Protocol

SUMMARIZE_PROMPT = """\
Summarize the following conversation turns into a structured format.

## Active Task
[What is currently being worked on]

## Goal
[The overall objective]

## Completed Actions
[What has been done]

## Key Decisions
[Important decisions made and why]

## Resolved Questions
[Questions that have been answered]

## Pending Questions
[Still open questions]

## Remaining Work
[What still needs to be done]

Do NOT answer questions or fulfill requests from the conversation. Only summarize.
"""

_PRUNED_PLACEHOLDER = "[Old tool output cleared to save context space]"


def count_tokens(text: str) -> int:
    """Rough token count. ~4 chars per token for English, ~1.5 chars for Chinese."""
    if not text:
        return 0
    cn_chars = sum(1 for c in text if '一' <= c <= '鿿')
    en_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + en_chars / 4)


def _msg_tokens(msg: dict) -> int:
    total = 0
    content = msg.get("content")
    if isinstance(content, str):
        total += count_tokens(content)
    elif content:
        total += count_tokens(str(content))
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            total += count_tokens(tc.get("function", {}).get("arguments", ""))
    return total


def _total_tokens(messages: list[dict]) -> int:
    return sum(_msg_tokens(m) for m in messages)


class LLMFn(Protocol):
    def __call__(self, prompt: str, max_tokens: int) -> str: ...


class MockLLM:
    """Returns a fixed-length summary for testing."""
    def __call__(self, prompt: str, max_tokens: int) -> str:
        return (
            "## Active Task\nTesting compression\n"
            "## Goal\nVerify 3-phase works\n"
            "## Completed Actions\n1. Phase 1 prune done\n2. Phase 2 protect done\n"
            "## Key Decisions\nUse structured summary\n"
            "## Resolved Questions\nHow to tokenize? -> rough estimate\n"
            "## Pending Questions\nNone\n"
            "## Remaining Work\nValidate with real conversations"
        )


class ContextCompressor:
    """3-phase context compression inspired by Hermes."""

    def __init__(
        self,
        llm_fn: LLMFn,
        context_window: int = 50000,
        threshold_percent: float = 0.50,
        summary_ratio: float = 0.20,
        min_summary_tokens: int = 2000,
        max_summary_tokens: int = 12000,
    ):
        self.llm_fn = llm_fn
        self.context_window = context_window
        self.threshold_percent = threshold_percent
        self.summary_ratio = summary_ratio
        self.min_summary_tokens = min_summary_tokens
        self.max_summary_tokens = max_summary_tokens
        self.threshold = int(context_window * threshold_percent)

    def needs_compression(self, messages: list[dict]) -> bool:
        """Check if total tokens exceed threshold_percent of context_window."""
        return _total_tokens(messages) >= self.threshold

    def compress(self, messages: list[dict]) -> tuple[list[dict], dict]:
        """Run 3-phase compression. Returns (compressed_messages, metrics).

        metrics = {original_tokens, compressed_tokens, compression_ratio,
                   phases: {pruned_count, protected_tokens, summary_tokens}}
        """
        original_tokens = _total_tokens(messages)
        # Phase 1
        pruned = self._phase_prune(messages)
        pruned_count = sum(
            1 for a, b in zip(messages, pruned) if a.get("content") != b.get("content")
        )
        # Phase 2
        budget = int(self.threshold * self.summary_ratio)
        protected, middle = self._phase_protect(pruned, budget)
        # Phase 3
        summary_budget = max(
            self.min_summary_tokens,
            min(int(_total_tokens(middle) * self.summary_ratio), self.max_summary_tokens),
        )
        summary_text = self._phase_summarize(middle, summary_budget)
        summary_msg = {"role": "user", "content": summary_text}
        # Assemble: head[0] (system) + summary + protected tail
        compressed = ([protected[0]] if protected else []) + [summary_msg] + protected[1:]
        compressed_tokens = _total_tokens(compressed)
        metrics = {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "compression_ratio": compressed_tokens / original_tokens if original_tokens else 0.0,
            "phases": {
                "pruned_count": pruned_count,
                "protected_tokens": _total_tokens(protected),
                "summary_tokens": _msg_tokens(summary_msg),
            },
        }
        return compressed, metrics

    def _phase_prune(self, messages: list[dict]) -> list[dict]:
        """Phase 1: Replace old tool outputs with 1-line placeholders."""
        result = []
        for msg in messages:
            m = msg.copy()
            content = m.get("content", "")
            if m.get("role") == "tool" and isinstance(content, str) and len(content) > 200:
                m["content"] = _PRUNED_PLACEHOLDER
            if m.get("role") == "assistant" and m.get("tool_calls"):
                new_tcs = []
                for tc in m["tool_calls"]:
                    tc = tc.copy()
                    fn = tc.get("function", {})
                    if isinstance(fn, dict) and len(fn.get("arguments", "")) > 500:
                        fn = fn.copy()
                        fn["arguments"] = fn["arguments"][:200] + "...[truncated]"
                        tc["function"] = fn
                    new_tcs.append(tc)
                m["tool_calls"] = new_tcs
            result.append(m)
        return result

    def _phase_protect(
        self, messages: list[dict], budget: int
    ) -> tuple[list[dict], list[dict]]:
        """Phase 2: Separate into protected (head system + recent tail) and middle.

        Protected = system prompt + recent messages fitting in ~60% of budget.
        Returns (protected_messages, middle_messages).
        """
        if not messages:
            return [], []
        head = [messages[0]]
        # Walk backward accumulating tokens up to 60% of budget
        tail_budget = int(budget * 0.6)
        tail_start = len(messages)
        accumulated = 0
        for i in range(len(messages) - 1, 0, -1):
            tokens = _msg_tokens(messages[i])
            if accumulated + tokens > tail_budget and (len(messages) - i) >= 2:
                break
            accumulated += tokens
            tail_start = i
        tail_start = max(tail_start, 1)
        return head + messages[tail_start:], messages[1:tail_start]

    def _phase_summarize(self, middle: list[dict], summary_budget: int) -> str:
        """Phase 3: LLM structured summary of middle section."""
        if not middle:
            return "[No middle turns to summarize]"
        parts: list[str] = []
        for msg in middle:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 6000:
                content = content[:4000] + "\n...[truncated]...\n" + content[-1500:]
            if role == "tool":
                parts.append(f"[TOOL RESULT]: {content}")
            elif role == "assistant":
                tool_info = ""
                for tc in msg.get("tool_calls") or []:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    args = fn.get("arguments", "")
                    if len(args) > 1500:
                        args = args[:1200] + "..."
                    tool_info += f"\n  {fn.get('name', '?')}({args})"
                parts.append(f"[ASSISTANT]: {content}{tool_info}")
            else:
                parts.append(f"[{role.upper()}]: {content}")
        prompt = f"{SUMMARIZE_PROMPT}\n\n---\nCONVERSATION:\n" + "\n\n".join(parts)
        return self.llm_fn(prompt, summary_budget)


_TOOL_NAMES = ["read_file", "terminal", "search_files", "write_file", "patch"]
_TOOL_TARGETS = ["src/app.py", "config.yaml", "tests/test_main.py", "README.md", "lib/utils.py"]


def generate_long_conversation(n_turns: int = 100, tokens_per_turn: int = 300) -> list[dict]:
    """Generate a long conversation with tool calls, decisions, and Q&A."""
    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful coding assistant."}
    ]
    filler = (
        "The implementation involves careful consideration of edge cases "
        "and proper error handling throughout the codebase. "
    )
    n_repeats = max(1, tokens_per_turn // count_tokens(filler))
    bulk = filler * n_repeats

    for i in range(n_turns):
        kind = random.choice(["user_question", "assistant_tool", "assistant_text", "tool_result"])
        target = random.choice(_TOOL_TARGETS)
        tool = random.choice(_TOOL_NAMES)
        if kind == "user_question":
            messages.append({"role": "user",
                "content": f"Turn {i}: Can you check {target} and explain? {bulk}"})
        elif kind == "assistant_tool":
            messages.append({"role": "assistant", "content": f"Let me look at {target}.",
                "tool_calls": [{"id": f"call_{i}", "type": "function",
                    "function": {"name": tool,
                        "arguments": f'{{"path": "{target}", "query": "review"}}'}}]})
        elif kind == "tool_result":
            messages.append({"role": "tool",
                "content": f"Result from {tool} on {target}:\n{bulk}",
                "tool_call_id": f"call_{i - 1}" if i > 0 else f"call_{i}"})
        else:
            messages.append({"role": "assistant",
                "content": f"Analysis of {target}: looks good. {bulk}"})
    return messages
