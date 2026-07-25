import pytest

from main import parse_args


def test_query_arguments():
    args = parse_args(["query", "hello", "world", "--top-k", "6", "--no-cache"])

    assert args.question == ["hello", "world"]
    assert args.top_k == 6
    assert args.no_cache is True


def test_command_is_required():
    with pytest.raises(SystemExit):
        parse_args([])
