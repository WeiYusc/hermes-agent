from __future__ import annotations

from unittest.mock import patch

from agent.context_compressor import ContextCompressor


def _compressor() -> ContextCompressor:
    with patch("agent.context_compressor.get_model_context_length", return_value=100_000):
        return ContextCompressor(
            model="test/model",
            threshold_percent=0.85,
            protect_first_n=1,
            protect_last_n=2,
            quiet_mode=True,
        )


def test_serialize_for_summary_with_budget_keeps_recent_turns_and_marks_omissions() -> None:
    compressor = _compressor()
    turns = [
        {"role": "tool", "tool_call_id": f"old-{i}", "content": f"OLD_{i}_" + ("x" * 80)}
        for i in range(8)
    ] + [
        {"role": "user", "content": "RECENT_USER_KEEP"},
        {"role": "assistant", "content": "RECENT_ASSISTANT_KEEP"},
    ]

    text = compressor._serialize_for_summary_with_budget(turns, max_chars=360)

    assert len(text) <= 360
    assert "earlier compacted turn(s) omitted" in text
    assert "RECENT_USER_KEEP" in text
    assert "RECENT_ASSISTANT_KEEP" in text
    assert "OLD_0_" not in text


def test_serialize_for_summary_with_budget_keeps_contiguous_recent_suffix() -> None:
    compressor = _compressor()
    turns = [
        {"role": "user", "content": "OLDER_SMALL_SHOULD_NOT_BACKFILL"},
        {"role": "tool", "tool_call_id": "large-middle", "content": "MIDDLE_TOO_LARGE_" + ("x" * 500)},
        {"role": "assistant", "content": "NEWEST_KEEP"},
    ]

    text = compressor._serialize_for_summary_with_budget(turns, max_chars=180)

    assert len(text) <= 180
    assert "earlier compacted turn(s) omitted" in text
    assert "NEWEST_KEEP" in text
    assert "MIDDLE_TOO_LARGE_" not in text
    assert "OLDER_SMALL_SHOULD_NOT_BACKFILL" not in text


def test_serialize_for_summary_without_budget_preserves_existing_behavior() -> None:
    compressor = _compressor()
    turns = [
        {"role": "assistant", "content": "Here is MEDIA:/tmp/audio.ogg"},
        {"role": "tool", "tool_call_id": "t1", "content": "Generated MEDIA:/tmp/out.mp3"},
    ]

    text = compressor._serialize_for_summary_with_budget(turns, max_chars=None)

    assert "MEDIA:" not in text
    assert text.count("[media attachment]") == 2
    assert "[ASSISTANT]: Here is [media attachment]" in text
    assert "[TOOL RESULT t1]: Generated [media attachment]" in text
