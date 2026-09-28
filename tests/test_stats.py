import gzip
import json
from pathlib import Path

from typer.testing import CliRunner

from pick.stats import typer_app

runner = CliRunner()

_CRAWL = {
    "stats": [
        {"event_id": 1, "team_a": [10, 11, 12], "team_b": [20, 21, 22], "a_wins": 6, "total": 10},
        {"event_id": 1, "team_a": [10, 11, 12], "team_b": [30, 31, 32], "a_wins": 4, "total": 10},
    ]
}


def _write_crawl(path: Path, gzipped: bool) -> Path:
    if gzipped:
        path = path.with_suffix(path.suffix + ".gz")
        with gzip.open(path, "wt") as f:
            json.dump(_CRAWL, f)
    else:
        path.write_text(json.dumps(_CRAWL))
    return path


class TestWinrates:
    def test_plain_input(self, tmp_path: Path):
        crawl_path = _write_crawl(tmp_path / "crawl_leg.json", gzipped=False)
        output = tmp_path / "winrates.json"
        result = runner.invoke(typer_app, ["winrates", "--input", str(crawl_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert json.loads(output.read_text())

    def test_gzipped_input_matches_plain(self, tmp_path: Path):
        plain_path = _write_crawl(tmp_path / "crawl_leg.json", gzipped=False)
        gz_path = _write_crawl(tmp_path / "crawl_leg2.json", gzipped=True)

        plain_out = tmp_path / "winrates_plain.json"
        gz_out = tmp_path / "winrates_gz.json"
        runner.invoke(typer_app, ["winrates", "--input", str(plain_path), "--output", str(plain_out)])
        result = runner.invoke(typer_app, ["winrates", "--input", str(gz_path), "--output", str(gz_out)])

        assert result.exit_code == 0, result.output
        assert json.loads(gz_out.read_text()) == json.loads(plain_out.read_text())


class TestPickrates:
    def test_gzipped_input(self, tmp_path: Path):
        crawl_path = _write_crawl(tmp_path / "crawl_leg.json", gzipped=True)
        output = tmp_path / "pickrates.json"
        result = runner.invoke(typer_app, ["pickrates", "--input", str(crawl_path), "--output", str(output)])
        assert result.exit_code == 0, result.output
        assert json.loads(output.read_text())
