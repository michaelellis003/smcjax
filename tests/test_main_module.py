# Copyright 2026 Michael Ellis
# SPDX-License-Identifier: Apache-2.0
from smcjax.__main__ import main


def test_main_prints_version(capsys):
    """Test that main prints the package name and version."""
    main()
    captured = capsys.readouterr()
    assert captured.out.startswith('smcjax ')
    assert len(captured.out.strip()) > len('smcjax ')
