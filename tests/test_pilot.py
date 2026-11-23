"""Pilot batches: pinned version/correction/offset, out-of-order reports,
report corrections, failure retention and model-version changes."""
import random

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.model import RidgeModel, mean_residual
from app.stats import Stats, apply_correction, mean_and_cov


def make_rows(n, speed_mu, press_mu, coef, intercept, noise, seed,
              press_shift=0, power_bias=0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = int(round(rng.gauss(speed_mu, 25)))
        p = int(round(rng.gauss(press_mu, 12)))
        y = (coef[0] * s + coef[1] * (p + press_shift) + intercept
             + power_bias + rng.gauss(0, noise))
        rows.append((s, p, int(round(y))))
    return rows


def payload(factory_id, rows, correction=0):
    n = len(rows)
    return {
        "factory_id": factory_id,
        "n": n,
        "sums": [sum(r[i] for r in rows) for i in range(3)],
        "outer": [[sum(r[i] * r[j] for r in rows) for j in range(3)]
                  for i in range(3)],
        "correction": correction,
    }


def summary(rows):
    """Trial-run report body: integer sufficient statistics only."""
    n = len(rows)
    return {
        "n": n,
        "sums": [sum(r[i] for r in rows) for i in range(3)],
        "outer": [[sum(r[i] * r[j] for r in rows) for j in range(3)]
                  for i in range(3)],
    }


def mean_residual_from_rows(rows, model, correction):
    """Reference offset: mean(actual power - shared prediction)."""
    return sum(
        y - model.predict(s, p + correction) for s, p, y in rows
    ) / len(rows)


def mse_from_rows(rows, model, correction, offset):
    return sum(
        (y - (model.predict(s, p + correction) + offset)) ** 2
        for s, p, y in rows
    ) / len(rows)


def demo_factories():
    rows = [
        make_rows(400, 1_000_000 + 30 * i, 500_000 + 8 * i, (0.02, 0.011),
                  1500, 4, seed=200 + i)
        for i in range(4)
    ]
    rows[2] = make_rows(400, 1_000_060, 500_016, (0.02, 0.011), 1500, 4,
                        seed=999, press_shift=25)
    return [payload(f"F{i+1}", r) for i, r in enumerate(rows)]


@pytest.fixture()
def client():
    app = create_app(":memory:")
    with TestClient(app) as c:
        yield c


def create(client, factories=None):
    r = client.post("/versions", json={"factories": factories or demo_factories()})
    assert r.status_code == 201, r.text
    return r.json()


def open_batch(client, factory_id, batch_id, expect=201):
    r = client.post("/pilot/batches",
                    json={"factory_id": factory_id, "batch_id": batch_id})
    assert r.status_code == expect, r.text
    return r.json()


def report(client, factory_id, batch_id, rows, expect=200):
    r = client.post(f"/pilot/batches/{factory_id}/{batch_id}/reports",
                    json=summary(rows))
    assert r.status_code == expect, r.text
    return r.json()


# --------------------------------------------------------------- opening
def test_open_batch_pins_current_version_and_offset_zero(client):
    ver = create(client)
    batch = open_batch(client, "F1", "B1")
    assert batch["version"] == ver["version"]
    assert batch["correction"] == ver["corrections"]["F1"]
    assert batch["offset"] == 0.0


def test_reopen_returns_original_pins_even_after_new_version(client):
    create(client)
    first = open_batch(client, "F1", "B1")
    # publish version 2 via a correction update
    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 25})
    assert r.status_code == 200
    assert client.get("/versions/current").json()["version"] == 2
    again = open_batch(client, "F1", "B1", expect=200)
    assert again["version"] == first["version"] == 1
    assert again["correction"] == first["correction"]
    assert again["offset"] == first["offset"]
    assert again["opened_at"] == first["opened_at"]


def test_open_unknown_factory_or_without_version(client):
    r = client.post("/pilot/batches",
                    json={"factory_id": "F1", "batch_id": "B1"})
    assert r.status_code == 503
    create(client)
    r = client.post("/pilot/batches",
                    json={"factory_id": "NOPE", "batch_id": "B1"})
    assert r.status_code == 404


def test_open_never_scans_individual_reports(client):
    create(client)
    open_batch(client, "F1", "B1")
    report(client, "F1", "B1", make_rows(120, 1_000_000, 500_000,
                                        (0.02, 0.011), 1500, 4, seed=1))
    # A second batch with a report replaced in between: offsets are derived
    # from the running totals row, not by walking pilot_reports.
    open_batch(client, "F1", "B2")
    report(client, "F1", "B2", make_rows(90, 1_000_000, 500_000,
                                        (0.02, 0.011), 1500, 4, seed=2))
    seen = []
    client.app.state.store._conn.set_trace_callback(seen.append)
    open_batch(client, "F1", "B3")
    client.app.state.store._conn.set_trace_callback(None)
    assert seen, "trace should capture the open transaction"
    assert not [q for q in seen if "pilot_reports" in q]
    assert [q for q in seen if "report_totals" in q]


# --------------------------------------------------------------- offsets
def test_offset_is_mean_residual_of_already_reported_samples(client):
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    corr = ver["corrections"]["F1"]

    open_batch(client, "F1", "B1")
    rows = make_rows(200, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                     seed=50, power_bias=40)
    rep = report(client, "F1", "B1", rows)
    # first batch had nothing to learn from -> offset 0
    assert rep["offset"] == 0.0
    assert rep["mse_before"] == pytest.approx(mse_from_rows(rows, model, corr, 0.0))
    assert rep["mse_after"] == pytest.approx(mse_from_rows(rows, model, corr, 0.0))

    batch2 = open_batch(client, "F1", "B2")
    want = mean_residual_from_rows(rows, model, corr)
    assert batch2["offset"] == pytest.approx(want)
    assert batch2["offset"] == pytest.approx(40.0, abs=1.0)


def test_predict_with_batch_uses_pinned_model_correction_offset(client):
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    open_batch(client, "F3", "B1")
    rows = make_rows(200, 1_000_060, 500_016, (0.02, 0.011), 1500, 4,
                     seed=61, press_shift=25, power_bias=30)
    report(client, "F3", "B1", rows)
    batch = open_batch(client, "F3", "B2")

    s, p, _ = rows[0]
    r = client.post("/predict", json={"factory_id": "F3", "batch_id": "B2",
                                      "speed": s, "pressure": p})
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == batch["version"] == 1
    assert body["correction"] == batch["correction"]
    assert body["offset"] == batch["offset"]
    assert body["power"] == pytest.approx(
        model.predict(s, p + batch["correction"]) + batch["offset"])


def test_predict_unknown_batch_and_report_unopened_batch(client):
    create(client)
    r = client.post("/predict", json={"factory_id": "F1", "batch_id": "NOPE",
                                      "speed": 1_000_000, "pressure": 500_000})
    assert r.status_code == 404
    r = client.post("/pilot/batches/F1/NOPE/reports",
                    json=summary([(1_000_000, 500_000, 21_500)] * 3))
    assert r.status_code == 404
    assert client.get("/pilot/batches/F1/NOPE").status_code == 404


# --------------------------------------------------------------- reports
def test_batches_report_out_of_order_with_own_pins(client):
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    open_batch(client, "F2", "A")
    open_batch(client, "F2", "B")
    rows_b = make_rows(90, 1_000_030, 500_008, (0.02, 0.011), 1500, 4,
                       seed=71, power_bias=15)
    rows_a = make_rows(110, 1_000_030, 500_008, (0.02, 0.011), 1500, 4,
                       seed=72, power_bias=-10)
    # no offset yet for either: both opened before any report
    rep_b = report(client, "F2", "B", rows_b)
    rep_a = report(client, "F2", "A", rows_a)
    corr = ver["corrections"]["F2"]
    for rep, rows in ((rep_b, rows_b), (rep_a, rows_a)):
        assert rep["version"] == 1
        assert rep["correction"] == corr
        assert rep["offset"] == 0.0
        assert rep["mse_after"] == pytest.approx(mse_from_rows(rows, model, corr, 0.0))

    # a third batch opened now sees both reports in the offset
    batch_c = open_batch(client, "F2", "C")
    pooled = rows_a + rows_b
    assert batch_c["offset"] == pytest.approx(
        mean_residual_from_rows(pooled, model, corr))


def test_report_correction_updates_batch_and_feeds_later_opens(client):
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    corr = ver["corrections"]["F1"]
    open_batch(client, "F1", "B1")
    rows_v1 = make_rows(100, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                        seed=81, power_bias=25)
    rep = report(client, "F1", "B1", rows_v1)
    batch2 = open_batch(client, "F1", "B2")  # opened after B1's report, offset ~25
    assert batch2["offset"] == pytest.approx(
        mean_residual_from_rows(rows_v1, model, corr))

    # replace B1's summary: errors change, and so do the totals
    rows_v2 = make_rows(130, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                        seed=82, power_bias=60)
    rep2 = report(client, "F1", "B1", rows_v2)
    assert rep2["replaced"] is True
    assert rep2["mse_before"] == pytest.approx(
        mse_from_rows(rows_v2, model, corr, 0.0))
    assert rep2["mse_before"] != pytest.approx(rep["mse_before"])
    stored = client.get("/pilot/batches/F1/B1").json()
    assert stored["report"]["n"] == len(rows_v2)
    assert stored["report"]["mse_before"] == pytest.approx(rep2["mse_before"])

    # new-open offset reflects the replacement, not the old summary
    batch3 = open_batch(client, "F1", "B3")
    assert batch3["offset"] == pytest.approx(
        mean_residual_from_rows(rows_v2, model, corr))
    # ...and the already opened batch keeps its pins
    assert client.get("/pilot/batches/F1/B2").json()["offset"] == batch2["offset"]


def test_failed_report_keeps_original_summary_and_totals(client):
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    corr = ver["corrections"]["F1"]
    open_batch(client, "F1", "B1")
    rows = make_rows(100, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                     seed=91, power_bias=20)
    good = report(client, "F1", "B1", rows)

    bad = summary(rows)
    bad["outer"][0][0] = 1  # impossible: sum of squares too small
    r = client.post("/pilot/batches/F1/B1/reports", json=bad)
    assert r.status_code == 422

    stored = client.get("/pilot/batches/F1/B1").json()["report"]
    assert stored["n"] == len(rows)
    assert stored["mse_before"] == pytest.approx(good["mse_before"])
    assert stored["mse_after"] == pytest.approx(good["mse_after"])
    after = open_batch(client, "F1", "B2")
    assert after["offset"] == pytest.approx(
        mean_residual_from_rows(rows, model, corr))


def test_store_failure_rolls_back_summary_and_totals(client, monkeypatch):
    """A write failure after the totals were touched leaves both untouched."""
    store = client.app.state.store
    create(client)
    open_batch(client, "F1", "B1")
    rows = make_rows(80, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                     seed=95, power_bias=10)
    good = report(client, "F1", "B1", rows)

    def totals_row():
        return store._conn.execute(
            "SELECT n, sums_json FROM report_totals WHERE factory_id='F1'"
        ).fetchone()

    before = tuple(totals_row())

    real_apply = store._apply_totals
    calls = {"n": 0}

    def boom(cur, factory_id, sign, n, sums):
        calls["n"] += 1
        if calls["n"] == 2:  # fail after the old summary was subtracted
            raise RuntimeError("disk full")
        return real_apply(cur, factory_id, sign, n, sums)

    monkeypatch.setattr(store, "_apply_totals", boom)
    noisy = summary(make_rows(50, 1_000_000, 500_000, (0.02, 0.011), 1500, 4,
                              seed=96, power_bias=90))
    with pytest.raises(RuntimeError):
        store.submit_report("F1", "B1", noisy["n"], noisy["sums"],
                            noisy["outer"], 1.0, 1.0)
    monkeypatch.setattr(store, "_apply_totals", real_apply)

    assert tuple(totals_row()) == before
    stored = client.get("/pilot/batches/F1/B1").json()["report"]
    assert stored["n"] == len(rows)
    assert stored["mse_before"] == pytest.approx(good["mse_before"])


# --------------------------------------------------------------- versioning
def test_old_batch_keeps_old_model_new_batch_uses_new(client):
    v1 = create(client)
    m1 = RidgeModel.from_dict(v1["model"])
    open_batch(client, "F3", "OLD")
    rows = make_rows(200, 1_000_060, 500_016, (0.02, 0.011), 1500, 4,
                     seed=101, press_shift=25, power_bias=35)
    old_rep = report(client, "F3", "OLD", rows)
    s, p, _ = rows[3]
    old_pred = client.post("/predict", json={
        "factory_id": "F3", "batch_id": "OLD", "speed": s, "pressure": p}
    ).json()

    # shared model is re-published: F3's correction changes 0 -> 25
    r = client.post("/versions/1/correction",
                    json={"factory_id": "F3", "correction": 25})
    assert r.status_code == 200
    v2 = r.json()
    m2 = RidgeModel.from_dict(v2["model"])

    # old batch: scoring and prediction still on version 1 pins
    old_again = report(client, "F3", "OLD", rows)
    assert old_again["version"] == 1
    assert old_again["mse_before"] == pytest.approx(old_rep["mse_before"])
    assert old_again["mse_after"] == pytest.approx(old_rep["mse_after"])
    pred_again = client.post("/predict", json={
        "factory_id": "F3", "batch_id": "OLD", "speed": s, "pressure": p}
    ).json()
    assert pred_again == old_pred

    # new batch: version 2 model + correction, offset from the old report
    # evaluated under version 2
    new = open_batch(client, "F3", "NEW")
    assert new["version"] == 2
    assert new["correction"] == 25
    assert new["offset"] == pytest.approx(
        mean_residual_from_rows(rows, m2, 25))
    assert new["offset"] != pytest.approx(
        mean_residual_from_rows(rows, m1, 0))

    rep_new = report(client, "F3", "NEW", rows)
    assert rep_new["version"] == 2
    assert rep_new["mse_after"] == pytest.approx(
        mse_from_rows(rows, m2, 25, new["offset"]))
    assert rep_new["mse_after"] < rep_new["mse_before"]


def test_open_during_recompute_uses_committed_version(client, monkeypatch):
    """Opening pins the last committed version, even mid-recomputation."""
    import threading

    import app.main as main_mod

    create(client)
    orig_train = main_mod.train
    started = threading.Event()
    release = threading.Event()

    def slow_train(stats):
        started.set()
        assert release.wait(10)
        return orig_train(stats)

    monkeypatch.setattr(main_mod, "train", slow_train)
    outcome = {}

    def do_update():
        outcome["r"] = client.post("/versions/1/correction",
                                   json={"factory_id": "F3",
                                         "correction": 500})

    t = threading.Thread(target=do_update)
    t.start()
    assert started.wait(10)
    batch = open_batch(client, "F3", "MID")
    assert batch["version"] == 1
    assert batch["correction"] == 0
    release.set()
    t.join(10)
    assert outcome["r"].status_code == 200
    assert client.get("/versions/current").json()["version"] == 2
    # the batch opened during the recomputation keeps its pins
    assert client.get("/pilot/batches/F3/MID").json()["version"] == 1


def test_offset_formula_matches_sufficient_statistics(client):
    """The offset is computed from pooled means, never from raw rows."""
    ver = create(client)
    model = RidgeModel.from_dict(ver["model"])
    rows = make_rows(70, 1_000_000, 500_000, (0.02, 0.011), 1500, 4, seed=111)
    body = summary(rows)
    st = Stats.build(body["n"], body["sums"], body["outer"])
    mean, _ = mean_and_cov(apply_correction(st, 7))
    got = mean_residual(mean, model)
    assert got == pytest.approx(mean_residual_from_rows(rows, model, 7))
