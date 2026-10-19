"""ASCII log banner alignment."""

from app.services.log_flair import framework_banner


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


def test_framework_banner_box_chars_aligned() -> None:
    text = framework_banner("Model task provisioning")
    lines = text.split("\n")
    assert lines[0][0] == "╔" and lines[1][0] == "║" and lines[2][0] == "╚"
    assert lines[0][-1] == "╗" and lines[1][-1] == "║" and lines[2][-1] == "╝"
    assert len(lines[0]) == len(lines[1]) == len(lines[2])
