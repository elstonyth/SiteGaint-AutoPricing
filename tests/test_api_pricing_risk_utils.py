import json
import logging
import os
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import requests

from src.pokedata_client.api_client import (
    PokedataClientError,
    PokedataApiKeyError,
    PokedataAuthError,
    PokedataClient,
    PokedataRateLimitError,
)
from src.pokedata_client.models import PokedataPriceInfo
from src.pricing.pricing_engine import (
    PricingEngine,
    attach_pricing,
    compute_new_price_myr,
)
from src.risk.threshold_engine import (
    ThresholdEngine,
    attach_pct_change_vs_last_run,
    compute_change_metrics,
    load_price_history,
    save_price_history,
)
from src.storage.settings_store import SettingsStore
from src.storage.stats_store import StatsStore
from src.utils.config_loader import AppConfig, load_config, load_env
from src.utils.io_helpers import (
    ensure_directory,
    get_file_list,
    read_csv_file,
    read_excel_file,
    validate_file_path,
    write_csv_file,
    write_excel_file,
)
from src.utils.logging_setup import JSONFormatter, setup_logging
from src.webapp.routes import get_mapping_info


class DummyFX:
    def __init__(self, rate: Decimal):
        self._rate = rate

    def get_rate(self) -> Decimal:
        return self._rate


class FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None, headers: dict | None = None):
        self.status_code = status_code
        self._body = body or {}
        self.headers = headers or {}
        self.text = json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if 400 <= self.status_code < 600:
            raise requests.exceptions.HTTPError(f"HTTP {self.status_code}")


class SequentialSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_pokedata_client_headers_and_auth_and_rate_limit(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="key-123")

    # Avoid real sleeps
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(client, "_reset_backoff", lambda: setattr(client, "_consecutive_429s", 0))

    session = SequentialSession([FakeResponse(200, {"ok": True})])
    client.session = session

    result = client._make_request("GET", "/hello")
    assert result["ok"] is True
    assert session.calls[0]["headers"]["Authorization"] == "Bearer key-123"

    # 401 should raise PokedataAuthError
    client.session = SequentialSession([FakeResponse(401, {})])
    with pytest.raises(PokedataAuthError):
        client._make_request("GET", "/fail-auth")

    # 429 should back off and then raise PokedataRateLimitError after retries
    rate_limit_calls = []

    def fake_handle(retry_after=None):
        rate_limit_calls.append(retry_after)
        import time as _time
        client._backoff_until = _time.time() + (retry_after or 1)

    monkeypatch.setattr(client, "_handle_rate_limit", fake_handle)
    client.session = SequentialSession([
        FakeResponse(429, {}, {"Retry-After": "1"}),
        FakeResponse(429, {}, {"Retry-After": "2"}),
        FakeResponse(429, {}, {"Retry-After": "3"}),
    ])
    with pytest.raises(PokedataRateLimitError):
        client._make_request("GET", "/rate")
    assert rate_limit_calls == [1, 2, 3]


def test_pokedata_client_product_and_verify(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="abc")

    monkeypatch.setattr(client, "_make_request", lambda *a, **k: {"id": 10, "name": "Prod", "language": "ENGLISH"})
    product = client.get_product("10")
    assert product.product_id == "10"
    assert product.name == "Prod"

    # verify_account returns False on auth error
    monkeypatch.setattr(client, "_make_request", lambda *a, **k: (_ for _ in ()).throw(PokedataAuthError("bad")))
    assert client.verify_account() is False

    # Missing API key should trigger error when enforced
    monkeypatch.delenv("POKEDATA_API_KEY", raising=False)
    empty_client = PokedataClient(config, api_key="")
    with pytest.raises(PokedataApiKeyError):
        empty_client.require_api_key()


def test_pricing_engine_batch_rounding_and_summary():
    config = AppConfig()
    config.rounding.decimal_places = 0
    config.rounding.method = "ceil"
    config.rounding.round_to_nearest = 1.0

    engine = PricingEngine(config, fx_provider=DummyFX(Decimal("4.0")))
    engine.set_margin_divisor(0.5)

    df = pd.DataFrame({"pokedata_price_usd": [2.5, None, -1, "bad"]})
    out = engine.calculate_prices_batch(df)
    assert out["new_price_myr"].tolist()[0] == 20.0  # (2.5*4)/0.5 = 20, ceil to 20
    assert pd.isna(out["new_price_myr"].tolist()[1])

    summary = engine.get_pricing_summary(5.0)
    assert "USD $5.00" in summary

    assert compute_new_price_myr(10.0, 4.0, margin_divisor=0.8, decimal_places=2) == 50.0


def test_attach_pricing_with_dict_and_object():
    config = AppConfig()
    df = pd.DataFrame({"sku": ["A", "B"], "pokedata_id": ["1", "2"], "price": [10.0, 20.0]})
    prices = {
        "1": PokedataPriceInfo(product_id="1", primary_price_usd=5.0),
        "2": {"primary_price_usd": 2.0},
    }
    out = attach_pricing(df, prices, fx_rate=4.0, config=config)
    assert out.loc[out["sku"] == "A", "new_price_myr"].iloc[0] == 25.0
    assert out.loc[out["sku"] == "B", "pct_change"].iloc[0] == -50.0


def test_threshold_engine_cache_and_history(tmp_path):
    engine = ThresholdEngine(AppConfig())
    prices = {"10": 1.5}
    cache_file = tmp_path / "cache.csv"
    engine.save_current_prices(prices, cache_file)
    engine.load_previous_prices(cache_file)
    assert engine.get_previous_price("10") == 1.5

    history_file = tmp_path / "history.csv"
    results = pd.DataFrame({"sku": ["X"], "pokedata_id": ["1"], "new_price_myr": [9.0]})
    save_price_history(results, history_file)
    history_df = load_price_history(history_file)
    augmented = attach_pct_change_vs_last_run(results, history_df)
    assert augmented["last_new_price_myr"].iloc[0] == 9.0
    assert augmented["pct_change_vs_last_run"].iloc[0] == 0.0

    metrics = compute_change_metrics(10.0, 12.0)
    assert metrics["delta"] == 2.0 and metrics["pct_change"] == 20.0


def test_settings_and_stats_store(tmp_path):
    settings_path = tmp_path / "settings.json"
    store = SettingsStore(settings_path=str(settings_path))
    store.set_api_key("token")
    # Use get_api_key() which decrypts the stored value
    assert store.get_api_key() == "token"
    store.clear_api_key()
    assert store.get_api_key() == ""

    stats_path = tmp_path / "stats.csv"
    stats = StatsStore(stats_path=str(stats_path))
    stats.record_run(products_processed=2, products_updated=1, ok_count=1, fx_rate=4.5)
    today_runs = stats.get_today_runs()
    assert len(today_runs) == 1
    dashboard = stats.get_dashboard_stats()
    assert dashboard.today_processed == 2


def test_io_helpers_and_logging(tmp_path):
    excel_path = tmp_path / "file.xlsx"
    csv_path = tmp_path / "file.csv"

    df = pd.DataFrame({"a": [1, 2]})
    write_excel_file(df, excel_path)
    read_back = read_excel_file(excel_path)
    assert list(read_back["a"]) == [1, 2]

    write_csv_file(df, csv_path)
    csv_back = read_csv_file(csv_path)
    assert csv_back.shape[0] == 2

    assert ensure_directory(tmp_path / "subdir").exists()
    assert validate_file_path(csv_path) is True
    assert get_file_list(tmp_path, pattern="*.csv") == [csv_path]

    formatter = JSONFormatter()
    record = logging.LogRecord("x", logging.INFO, __file__, 10, "hello", args=(), exc_info=None, func="f")
    formatted = json.loads(formatter.format(record))
    assert formatted["message"] == "hello"

    log_file = tmp_path / "app.log"
    logger = setup_logging(level=logging.INFO, log_file=log_file, json_format=True)
    logger.info("test-log")
    assert log_file.exists()


def test_threshold_evaluate_statuses():
    engine = ThresholdEngine(AppConfig())
    df = pd.DataFrame(
        {
            "is_mapped": [False, True, True, True],
            "new_price_myr": [None, 12.0, 20.0, 5.0],
            "pct_change": [None, 30.0, 60.0, None],
            "abs_change": [None, 3.0, 10.0, 0.5],
        }
    )
    evaluated = engine.evaluate_batch(df)
    assert evaluated.loc[0, "status"] == "UNMAPPED"
    assert evaluated.loc[1, "status"] == "WARNING"
    assert evaluated.loc[2, "status"] == "BLOCKED"
    assert evaluated.loc[3, "status"] == "OK"


def test_pokedata_client_error_paths(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="key-err")
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(client, "_reset_backoff", lambda: None)

    # 500 with json message -> auth error
    client.session = SequentialSession([FakeResponse(500, {"message": "auth fail"})])
    with pytest.raises(PokedataAuthError):
        client._make_request("GET", "/err")

    # Timeout and connection errors
    class TimeoutSession:
        def request(self, **kwargs):
            raise requests.exceptions.Timeout("timeout")

    client.session = TimeoutSession()
    with pytest.raises(PokedataClientError):
        client._make_request("GET", "/timeout")

    class ConnSession:
        def request(self, **kwargs):
            raise requests.exceptions.ConnectionError("conn")

    client.session = ConnSession()
    with pytest.raises(PokedataClientError):
        client._make_request("GET", "/conn")


def test_get_pricing_parses_sources(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="key")
    monkeypatch.setattr(client, "_record_price_history", lambda *a, **k: None)
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(client, "_reset_backoff", lambda: None)
    # Disable cache to get fresh results
    client.set_use_cache(False)

    pricing_payload = {
        "pricing": {
            "Pokedata Raw": {"currency": "usd", "value": 5},
            "TCGPlayer": {"currency": "usd", "value": 10},
        }
    }
    client.session = SequentialSession([FakeResponse(200, pricing_payload)])
    info = client.get_pricing("123")
    assert info.primary_price_usd == 5
    assert info.source == "Pokedata Raw"

    # Fallback to average when no priority price
    payload_avg = {"pricing": {"Other": {"value": 4}, "Another": {"value": 6}}}
    client.session = SequentialSession([FakeResponse(200, payload_avg)])
    info2 = client.get_pricing("123")
    assert info2.primary_price_usd == 5
    assert info2.source == "average"


def test_search_products_and_price_wrappers(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="key")
    monkeypatch.setattr(client, "_make_request", lambda *a, **k: {"results": [{"id": 1, "name": "A"}]})
    results = client.search_products("A")
    assert results[0].product_id == "1"

    # get_price wrapper returns PokedataPriceData when price exists
    monkeypatch.setattr(client, "get_pricing", lambda pid: PokedataPriceInfo(product_id=pid, primary_price_usd=2.0))
    price_data = client.get_price("1")
    assert price_data.price_usd == 2.0

    # list response handling
    monkeypatch.setattr(client, "_make_request", lambda *a, **k: [{"id": 2, "name": "B"}])
    results_list = client.search_products("B")
    assert results_list[0].product_id == "2"

    # get_price returns None when no price
    monkeypatch.setattr(client, "get_pricing", lambda pid: PokedataPriceInfo(product_id=pid, primary_price_usd=None))
    assert client.get_price("1") is None


def test_get_prices_batch(monkeypatch):
    config = AppConfig()
    client = PokedataClient(config, api_key="key")
    monkeypatch.setattr(client, "_rate_limit", lambda: None)
    monkeypatch.setattr(client, "_reset_backoff", lambda: None)
    # Mock get_pricing to accept force_refresh kwarg
    monkeypatch.setattr(client, "get_pricing", lambda pid, force_refresh=False: PokedataPriceInfo(product_id=pid, primary_price_usd=float(pid)))
    # Disable cache so batch fetches from API
    client.set_use_cache(False)

    results = client.get_prices_batch(["1", "2"], max_workers=2)
    assert results["1"].primary_price_usd == 1.0
    assert results["2"].primary_price_usd == 2.0


def test_config_loader_and_mapping_info(tmp_path, monkeypatch):
    # load_env when file missing
    load_env(tmp_path / "none.env")

    # load_config with existing file
    cfg_file = tmp_path / "config.yaml"
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    cfg_file.write_text("paths:\\n  input_dir: data/input\\n")
    cfg = load_config(cfg_file)
    assert isinstance(cfg, AppConfig)

    missing_cfg = load_config(tmp_path / "missing.yaml")
    assert isinstance(missing_cfg, AppConfig)

    env_file = tmp_path / ".env"
    env_file.write_text("HELLO=WORLD\n")
    load_env(env_file)
    assert os.environ.get("HELLO") == "WORLD"

    # get_mapping_info with csv
    mapping_path = tmp_path / "map.csv"
    mapping_path.write_text("sku,pokedata_id\\nA,1\\n")
    info = get_mapping_info(mapping_path)
    assert info["exists"] is True
    assert info.get("row_count", 0) >= 0


def test_io_helpers_errors_and_logging(tmp_path, capsys):
    # read_excel unsupported extension
    bad_file = tmp_path / "file.txt"
    bad_file.write_text("x")
    with pytest.raises(ValueError):
        read_excel_file(bad_file)

    # validate_file_path false cases
    assert validate_file_path(tmp_path / "missing.csv", must_exist=True) is False
    assert validate_file_path(tmp_path / "missing.csv", must_exist=False) is True

    # get_logger helper
    log_path = tmp_path / "log.log"
    logger = setup_logging(log_file=log_path)
    child = logging.getLogger("child")
    child.info("hello")
    assert log_path.exists()
