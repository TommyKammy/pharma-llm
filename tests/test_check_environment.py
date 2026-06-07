from scripts.check_environment import node_version_ok


def test_node_version_ok_accepts_promptfoo_supported_versions() -> None:
    assert node_version_ok("v20.20.0")
    assert node_version_ok("v22.22.0")
    assert node_version_ok("v25.5.0")


def test_node_version_ok_rejects_older_versions() -> None:
    assert not node_version_ok("v20.19.9")
    assert not node_version_ok("v18.20.0")
    assert not node_version_ok("not-a-version")
