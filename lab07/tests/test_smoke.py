from pathlib import Path


def test_blocked_variant_notice_exists():
    root = Path(__file__).parents[1]
    assert (root / "BLOCKED_BY_VARIANT_DATA.md").exists()
    assert not list(root.glob("Var_*.txt"))
