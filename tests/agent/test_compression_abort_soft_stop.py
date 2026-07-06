from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from hermes_state import SessionDB


class _AbortCompressor:
    protect_first_n = 0
    protect_last_n = 0
    threshold_tokens = 1
    context_length = 100_000
    last_real_prompt_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0
    compression_count = 0
    _last_summary_error = "Request timed out"
    _last_compress_aborted = False
    _last_aux_model_failure_model = None
    _last_aux_model_failure_error = None

    def should_compress(self, _tokens):
        return True

    def should_defer_preflight_to_real_usage(self, _tokens):
        return False

    def get_active_compression_failure_cooldown(self):
        return None

    def compress(self, messages, *_, **__):
        self._last_compress_aborted = True
        self._last_summary_error = "Request timed out"
        return messages


def _build_agent(db: SessionDB, session_id: str, tmp_path: Path):
    from run_agent import AIAgent

    agent = AIAgent(
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
        model="test/model",
        quiet_mode=True,
        session_db=db,
        session_id=session_id,
        skip_context_files=True,
        skip_memory=True,
    )
    agent.context_compressor = _AbortCompressor()
    agent.compression_enabled = True
    agent.tools = []
    agent.max_iterations = 5
    agent._emit_status = lambda *_a, **_k: None
    agent._emit_warning = lambda *_a, **_k: None
    agent._build_system_prompt = lambda _system_message=None: "sys"
    agent._cached_system_prompt = "sys"
    agent._cleanup_dead_connections = lambda: False
    agent._memory_manager = None
    agent._todo_store = SimpleNamespace(
        has_items=lambda: False,
        format_for_injection=lambda: "- pending: resume deployment",
    )
    return agent


def test_preflight_compression_abort_soft_stops_before_api_call(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "ABORT_SOFT_STOP"
    db.create_session(sid, source="wecom")
    agent = _build_agent(db, sid, tmp_path)

    history = [
        {"role": "user", "content": "previous /www/temp/worklog.md host 103.231.56.143"},
        {"role": "assistant", "content": "ok"},
    ]
    with patch("agent.turn_context.estimate_request_tokens_rough", return_value=999_999):
        result = agent.run_conversation("continue with password=supersecret", conversation_history=history)

    assert result["compression_aborted"] is True
    assert result["api_calls"] == 0
    assert "Compression aborted" in result["final_response"]
    assert "Recovery checkpoint:" in result["final_response"]
    assert agent._last_compression_abort_should_stop is True

    checkpoint = Path(agent._last_compression_abort_checkpoint)
    assert checkpoint.exists()
    text = checkpoint.read_text()
    assert "Request timed out" in text
    assert "/www/temp/worklog.md" in text
    assert "103.231.56.143" in text
    assert "supersecret" not in text


def test_summary_template_contains_recovery_index() -> None:
    source = Path(__file__).parents[2] / "agent" / "context_compressor.py"
    text = source.read_text()
    assert "## Recovery Index" in text
    assert "Credential locations only" in text
    assert "Do-not-assume notes" in text
