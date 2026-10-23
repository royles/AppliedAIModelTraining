"""ASCII log banner alignment."""

from app.services.log_flair import (
    framework_banner,
    heartbeat_line,
    task_done_line,
    task_start_line,
)


def test_framework_banner_rows_equal_width() -> None:
    for title in (
        "Model task provisioning",
        "Hi",
        "Queue",
        "A very long title that might wrap the box width",
    ):
        lines = framework_banner(title).split("\n")
        assert len(lines) == 3
        widths = [len(line) for line in lines]
        assert widths[0] == widths[1] == widths[2], (title, widths, lines)


def test_heartbeat_line() -> None:
    assert "still running (45s)" in heartbeat_line("BPE tokenizer", 45)
    assert "download 5,000" in heartbeat_line("Corpus load", 30, "download 5,000/10,000 rows")


def test_task_phase_lines() -> None:
    assert "started" in task_start_line("train", "Dataset sharding")
    done = task_done_line("Dataset sharding", seconds=12.0, detail="100 ex → 4 shards")
    assert "12s" in done or "(12" in done
    assert "100 ex" in done


def test_framework_banner_box_chars_aligned() -> None:
    text = framework_banner("Model task provisioning")
    lines = text.split("\n")
    assert lines[0][0] == "╔" and lines[1][0] == "║" and lines[2][0] == "╚"
    assert lines[0][-1] == "╗" and lines[1][-1] == "║" and lines[2][-1] == "╝"
    assert len(lines[0]) == len(lines[1]) == len(lines[2])
