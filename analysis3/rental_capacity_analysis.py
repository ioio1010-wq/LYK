from __future__ import annotations

import html as html_lib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "LYK_Jejuro_data_package_2026-08-11" / "data"
VEH_DIR = ROOT / "LYK_Jejuro_vehicle_dataset_CSV_2026-08-18"
OUT_DIR = ROOT / "analysis3"

ANALYSIS_START = pd.Timestamp("2025-01-01")
ANALYSIS_END = pd.Timestamp("2026-08-11")
EXCESS_REPAIR_QUANTILE = 0.99

VALID_STATUSES = {"RETURNED", "CONFIRMED", "PENDING", "IN_USE", "IN_PROGRESS"}


@dataclass(frozen=True)
class Interval:
    model_name: str
    start: pd.Timestamp
    end: pd.Timestamp
    hours: float
    revenue: float


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def date_range_half_open(start: pd.Timestamp, end: pd.Timestamp) -> Iterable[pd.Timestamp]:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return []
    first = start.normalize()
    last = (end - pd.Timedelta(microseconds=1)).normalize()
    return pd.date_range(first, last, freq="D")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reservations = read_csv(RES_DIR / "reservations.csv")
    pre_reservations = read_csv(RES_DIR / "pre_reservations.csv")
    package_models = read_csv(RES_DIR / "vehicle_models.csv")
    vehicle_master = read_csv(VEH_DIR / "vehicle_master.csv")
    maintenance = read_csv(VEH_DIR / "maintenance_vehicle_summary.csv")

    reservations = reservations.merge(
        package_models[["vehicle_model_key", "vehicle_model_name", "vehicle_class"]],
        on="vehicle_model_key",
        how="left",
    )
    pre_reservations = pre_reservations.merge(
        package_models[["vehicle_model_key", "vehicle_model_name", "vehicle_class"]],
        on="vehicle_model_key",
        how="left",
    )

    return reservations, pre_reservations, vehicle_master, maintenance


def clean_reservations(reservations: pd.DataFrame) -> pd.DataFrame:
    df = reservations.copy()
    df["pickup_at"] = parse_dt(df["pickup_at_source"])
    df["dropoff_at"] = parse_dt(df["dropoff_at_source"])
    df["final_price_krw"] = pd.to_numeric(df["final_price_krw"], errors="coerce").fillna(0)
    df["rental_duration_hours"] = pd.to_numeric(df["rental_duration_hours"], errors="coerce")
    mask = (
        df["reservation_status"].isin(VALID_STATUSES)
        & ~df["is_deleted"].astype(bool)
        & df["pickup_at"].notna()
        & df["dropoff_at"].notna()
        & (df["dropoff_at"] > df["pickup_at"])
        & (df["dropoff_at"] > ANALYSIS_START)
        & (df["pickup_at"] < ANALYSIS_END + pd.Timedelta(days=1))
    )
    df = df.loc[mask].copy()
    df["start"] = df["pickup_at"].clip(lower=ANALYSIS_START)
    df["end"] = df["dropoff_at"].clip(upper=ANALYSIS_END + pd.Timedelta(days=1))
    df["hours_in_period"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df = df[df["hours_in_period"] > 0].copy()
    return df


def clean_pre_reservations(pre_reservations: pd.DataFrame) -> pd.DataFrame:
    df = pre_reservations.copy()
    df["pickup_at"] = parse_dt(df["pickup_at_source"])
    df["dropoff_at"] = parse_dt(df["dropoff_at_source"])
    df["price_krw"] = pd.to_numeric(df["price_krw"], errors="coerce").fillna(0)
    mask = (
        (df["checkout_outcome"] == "NOT_BOOKED_AS_OF_REFRESH")
        & ~df["is_deleted"].astype(bool)
        & df["pickup_at"].notna()
        & df["dropoff_at"].notna()
        & (df["dropoff_at"] > df["pickup_at"])
        & (df["dropoff_at"] > ANALYSIS_START)
        & (df["pickup_at"] < ANALYSIS_END + pd.Timedelta(days=1))
    )
    df = df.loc[mask].copy()
    df["start"] = df["pickup_at"].clip(lower=ANALYSIS_START)
    df["end"] = df["dropoff_at"].clip(upper=ANALYSIS_END + pd.Timedelta(days=1))
    df["hours_in_period"] = (df["end"] - df["start"]).dt.total_seconds() / 3600
    df = df[df["hours_in_period"] > 0].copy()
    return df


def build_daily_fleet(vehicle_master: pd.DataFrame) -> pd.DataFrame:
    vehicles = vehicle_master.copy()
    vehicles["fleet_start"] = pd.to_datetime(
        vehicles["vehicle_registration_date_observed"], errors="coerce"
    )
    fallback_start = parse_dt(vehicles["erp_record_created_at_utc"])
    vehicles["fleet_start"] = vehicles["fleet_start"].fillna(fallback_start)
    vehicles["fleet_end"] = parse_dt(vehicles["erp_soft_deleted_at_utc"])
    vehicles["fleet_end"] = vehicles["fleet_end"].fillna(ANALYSIS_END + pd.Timedelta(days=1))
    vehicles["fleet_start"] = vehicles["fleet_start"].clip(lower=ANALYSIS_START)
    vehicles["fleet_end"] = vehicles["fleet_end"].clip(upper=ANALYSIS_END + pd.Timedelta(days=1))
    vehicles = vehicles[
        vehicles["model_name"].notna()
        & vehicles["fleet_start"].notna()
        & vehicles["fleet_end"].notna()
        & (vehicles["fleet_end"] > vehicles["fleet_start"])
    ].copy()

    rows: list[dict[str, object]] = []
    for row in vehicles.itertuples(index=False):
        for day in date_range_half_open(row.fleet_start, row.fleet_end):
            rows.append(
                {
                    "date": day,
                    "model_name": row.model_name,
                    "vehicle_class": row.vehicle_class,
                    "vehicle_id": row.vehicle_id,
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame(columns=["date", "model_name", "vehicle_class", "fleet_count"])
    return (
        daily.groupby(["date", "model_name", "vehicle_class"], as_index=False)
        .agg(fleet_count=("vehicle_id", "nunique"))
        .sort_values(["date", "model_name"])
    )


def build_daily_usage(intervals_df: pd.DataFrame, price_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in intervals_df.itertuples(index=False):
        model_name = row.vehicle_model_name
        vehicle_class = row.vehicle_class
        if pd.isna(model_name):
            continue
        start = row.start
        end = row.end
        price = float(getattr(row, price_col))
        total_hours = max((end - start).total_seconds() / 3600, 0)
        if total_hours <= 0:
            continue
        revenue_per_hour = price / total_hours if total_hours else 0
        for day in date_range_half_open(start, end):
            day_start = day
            day_end = day + pd.Timedelta(days=1)
            overlap_start = max(start, day_start)
            overlap_end = min(end, day_end)
            hours = max((overlap_end - overlap_start).total_seconds() / 3600, 0)
            if hours <= 0:
                continue
            rows.append(
                {
                    "date": day,
                    "model_name": model_name,
                    "vehicle_class": vehicle_class,
                    "booked_hours": hours,
                    "revenue_allocated_krw": revenue_per_hour * hours,
                    "interval_start": overlap_start,
                    "interval_end": overlap_end,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "model_name",
                "vehicle_class",
                "booked_hours",
                "revenue_allocated_krw",
                "peak_concurrent",
            ]
        )
    expanded = pd.DataFrame(rows)
    daily = (
        expanded.groupby(["date", "model_name", "vehicle_class"], as_index=False)
        .agg(
            booked_hours=("booked_hours", "sum"),
            revenue_allocated_krw=("revenue_allocated_krw", "sum"),
        )
    )
    peaks = build_daily_peak_concurrency(expanded)
    return daily.merge(peaks, on=["date", "model_name", "vehicle_class"], how="left")


def build_daily_peak_concurrency(expanded: pd.DataFrame) -> pd.DataFrame:
    peaks: list[dict[str, object]] = []
    for key, group in expanded.groupby(["date", "model_name", "vehicle_class"], sort=False):
        events: list[tuple[pd.Timestamp, int]] = []
        for row in group.itertuples(index=False):
            events.append((row.interval_start, 1))
            events.append((row.interval_end, -1))
        current = 0
        peak = 0
        for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
            current += delta
            peak = max(peak, current)
        peaks.append(
            {
                "date": key[0],
                "model_name": key[1],
                "vehicle_class": key[2],
                "peak_concurrent": peak,
            }
        )
    return pd.DataFrame(peaks)


def consecutive_periods(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    stockout = daily[daily["is_stockout"]].sort_values(["model_name", "date"])
    for (model_name, vehicle_class), group in stockout.groupby(["model_name", "vehicle_class"]):
        dates = list(group["date"])
        if not dates:
            continue
        start = prev = dates[0]
        for day in dates[1:]:
            if day == prev + pd.Timedelta(days=1):
                prev = day
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "vehicle_class": vehicle_class,
                    "period_start": start,
                    "period_end": prev,
                    "stockout_days": (prev - start).days + 1,
                }
            )
            start = prev = day
        rows.append(
            {
                "model_name": model_name,
                "vehicle_class": vehicle_class,
                "period_start": start,
                "period_end": prev,
                "stockout_days": (prev - start).days + 1,
            }
        )
    return pd.DataFrame(rows)


def model_summary(daily: pd.DataFrame, periods: pd.DataFrame) -> pd.DataFrame:
    base = (
        daily.groupby(["model_name", "vehicle_class"], as_index=False)
        .agg(
            fleet_vehicle_days=("fleet_count", "sum"),
            booked_hours=("booked_hours", "sum"),
            revenue_krw=("revenue_allocated_krw", "sum"),
            observed_days=("date", "nunique"),
            stockout_days=("is_stockout", "sum"),
            max_peak_concurrent=("peak_concurrent", "max"),
            avg_fleet_count=("fleet_count", "mean"),
            avg_daily_peak=("peak_concurrent", "mean"),
        )
    )
    base["fleet_hours"] = base["fleet_vehicle_days"] * 24
    base["booked_vehicle_days"] = base["booked_hours"] / 24
    base["utilization_rate"] = np.where(
        base["fleet_vehicle_days"] > 0,
        base["booked_vehicle_days"] / base["fleet_vehicle_days"],
        np.nan,
    )
    base["stockout_day_rate"] = np.where(
        base["observed_days"] > 0, base["stockout_days"] / base["observed_days"], np.nan
    )
    base["revenue_per_booked_hour"] = np.where(
        base["booked_hours"] > 0, base["revenue_krw"] / base["booked_hours"], np.nan
    )
    if periods.empty:
        base["stockout_period_count"] = 0
        base["avg_stockout_period_days"] = 0
        base["max_stockout_period_days"] = 0
    else:
        per = (
            periods.groupby(["model_name", "vehicle_class"], as_index=False)
            .agg(
                stockout_period_count=("stockout_days", "count"),
                avg_stockout_period_days=("stockout_days", "mean"),
                max_stockout_period_days=("stockout_days", "max"),
            )
        )
        base = base.merge(per, on=["model_name", "vehicle_class"], how="left")
        base[["stockout_period_count", "avg_stockout_period_days", "max_stockout_period_days"]] = (
            base[["stockout_period_count", "avg_stockout_period_days", "max_stockout_period_days"]]
            .fillna(0)
        )
    return base.sort_values(["stockout_days", "revenue_krw"], ascending=[False, False])


def class_summary(summary: pd.DataFrame) -> pd.DataFrame:
    out = (
        summary.groupby("vehicle_class", as_index=False)
        .agg(
            model_count=("model_name", "nunique"),
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_hours=("booked_hours", "sum"),
            revenue_krw=("revenue_krw", "sum"),
            stockout_days=("stockout_days", "sum"),
            stockout_period_count=("stockout_period_count", "sum"),
        )
        .sort_values("revenue_krw", ascending=False)
    )
    out["fleet_hours"] = out["fleet_vehicle_days"] * 24
    out["utilization_rate"] = np.where(out["fleet_hours"] > 0, out["booked_hours"] / out["fleet_hours"], np.nan)
    return out


def repair_cost_by_model(daily: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    maint = maintenance.copy()
    maint["repair_cost_raw_krw"] = (
        pd.to_numeric(maint["maintenance_cost_krw_sum_observed"], errors="coerce").fillna(0)
        + pd.to_numeric(maint["detail_part_price_krw_sum_observed"], errors="coerce").fillna(0)
    )
    cap = maint["repair_cost_raw_krw"].quantile(EXCESS_REPAIR_QUANTILE)
    maint["repair_cost_winsorized_krw"] = maint["repair_cost_raw_krw"].clip(upper=cap)
    maint["repair_cost_removed_as_excess_krw"] = (
        maint["repair_cost_raw_krw"] - maint["repair_cost_winsorized_krw"]
    )
    repair = (
        maint.groupby(["model_name", "vehicle_class"], as_index=False)
        .agg(
            repair_cost_raw_krw=("repair_cost_raw_krw", "sum"),
            repair_cost_winsorized_krw=("repair_cost_winsorized_krw", "sum"),
            repair_cost_removed_as_excess_krw=("repair_cost_removed_as_excess_krw", "sum"),
        )
    )
    fleet_days = (
        daily.groupby(["model_name", "vehicle_class"], as_index=False)
        .agg(fleet_vehicle_days=("fleet_count", "sum"))
    )
    repair = repair.merge(fleet_days, on=["model_name", "vehicle_class"], how="right")
    for col in [
        "repair_cost_raw_krw",
        "repair_cost_winsorized_krw",
        "repair_cost_removed_as_excess_krw",
    ]:
        repair[col] = repair[col].fillna(0)
    repair["repair_cost_outlier_cap_vehicle_level_krw"] = cap
    repair["repair_cost_per_fleet_hour"] = np.where(
        repair["fleet_vehicle_days"] > 0,
        repair["repair_cost_winsorized_krw"] / (repair["fleet_vehicle_days"] * 24),
        0,
    )
    return repair


def model_profit_comparison(daily: pd.DataFrame, maintenance: pd.DataFrame) -> pd.DataFrame:
    revenue = (
        daily.groupby(["model_name", "vehicle_class"], as_index=False)
        .agg(
            fleet_vehicle_days=("fleet_count", "sum"),
            booked_hours=("booked_hours", "sum"),
            booked_revenue_krw=("revenue_allocated_krw", "sum"),
            stockout_days=("is_stockout", "sum"),
        )
    )
    revenue["booked_vehicle_days"] = revenue["booked_hours"] / 24
    revenue["utilization_rate"] = np.where(
        revenue["fleet_vehicle_days"] > 0,
        revenue["booked_vehicle_days"] / revenue["fleet_vehicle_days"],
        np.nan,
    )
    repair = repair_cost_by_model(daily, maintenance)
    out = revenue.merge(
        repair[
            [
                "model_name",
                "vehicle_class",
                "repair_cost_raw_krw",
                "repair_cost_winsorized_krw",
                "repair_cost_removed_as_excess_krw",
                "repair_cost_outlier_cap_vehicle_level_krw",
            ]
        ],
        on=["model_name", "vehicle_class"],
        how="left",
    )
    for col in [
        "repair_cost_raw_krw",
        "repair_cost_winsorized_krw",
        "repair_cost_removed_as_excess_krw",
    ]:
        out[col] = out[col].fillna(0)
    out["depreciation_cost_krw"] = np.nan
    out["depreciation_status"] = "unavailable_purchase_sale_fields"
    out["profit_proxy_before_depreciation_krw"] = (
        out["booked_revenue_krw"] - out["repair_cost_winsorized_krw"]
    )
    out["profit_margin_proxy_before_depreciation"] = np.where(
        out["booked_revenue_krw"] > 0,
        out["profit_proxy_before_depreciation_krw"] / out["booked_revenue_krw"],
        np.nan,
    )
    return out.sort_values("profit_proxy_before_depreciation_krw", ascending=False)


def incremental_opportunity(
    daily: pd.DataFrame,
    pre_reservations: pd.DataFrame,
    maintenance: pd.DataFrame,
) -> pd.DataFrame:
    latent = build_daily_usage(pre_reservations, "price_krw")
    if latent.empty:
        latent = pd.DataFrame(
            columns=["date", "model_name", "vehicle_class", "latent_hours", "latent_revenue_krw"]
        )
    else:
        latent = latent.rename(
            columns={
                "booked_hours": "latent_hours",
                "revenue_allocated_krw": "latent_revenue_krw",
                "peak_concurrent": "latent_peak_concurrent",
            }
        )

    opportunity = daily[daily["is_stockout"]].merge(
        latent[
            [
                "date",
                "model_name",
                "vehicle_class",
                "latent_hours",
                "latent_revenue_krw",
                "latent_peak_concurrent",
            ]
        ],
        on=["date", "model_name", "vehicle_class"],
        how="left",
    )
    for col in ["latent_hours", "latent_revenue_krw", "latent_peak_concurrent"]:
        opportunity[col] = pd.to_numeric(opportunity[col], errors="coerce").fillna(0)

    model_rate = (
        daily.groupby(["model_name", "vehicle_class"], as_index=False)
        .agg(booked_hours=("booked_hours", "sum"), revenue_krw=("revenue_allocated_krw", "sum"))
    )
    model_rate["revenue_per_booked_hour"] = np.where(
        model_rate["booked_hours"] > 0,
        model_rate["revenue_krw"] / model_rate["booked_hours"],
        np.nan,
    )

    repair_by_model = repair_cost_by_model(daily, maintenance)

    opportunity = opportunity.merge(model_rate, on=["model_name", "vehicle_class"], how="left")
    opportunity = opportunity.merge(
        repair_by_model[["model_name", "vehicle_class", "repair_cost_per_fleet_hour"]],
        on=["model_name", "vehicle_class"],
        how="left",
    )
    opportunity["repair_cost_per_fleet_hour"] = opportunity["repair_cost_per_fleet_hour"].fillna(0)
    opportunity["revenue_per_booked_hour"] = opportunity["revenue_per_booked_hour"].fillna(0)

    for extra in [1, 2]:
        hours_col = f"extra_{extra}_cars_absorbable_hours"
        revenue_col = f"extra_{extra}_cars_revenue_proxy_krw"
        repair_col = f"extra_{extra}_cars_repair_cost_proxy_krw"
        profit_col = f"extra_{extra}_cars_profit_proxy_krw"
        opportunity[hours_col] = np.minimum(opportunity["latent_hours"], 24 * extra)
        opportunity[revenue_col] = opportunity[hours_col] * opportunity["revenue_per_booked_hour"]
        opportunity[repair_col] = opportunity[hours_col] * opportunity["repair_cost_per_fleet_hour"]
        opportunity[profit_col] = opportunity[revenue_col] - opportunity[repair_col]

    cols = [
        "model_name",
        "vehicle_class",
        "latent_hours",
        "latent_revenue_krw",
        "extra_1_cars_absorbable_hours",
        "extra_1_cars_revenue_proxy_krw",
        "extra_1_cars_repair_cost_proxy_krw",
        "extra_1_cars_profit_proxy_krw",
        "extra_2_cars_absorbable_hours",
        "extra_2_cars_revenue_proxy_krw",
        "extra_2_cars_repair_cost_proxy_krw",
        "extra_2_cars_profit_proxy_krw",
    ]
    out = opportunity.groupby(["model_name", "vehicle_class"], as_index=False)[cols[2:]].sum()
    return out.sort_values("extra_1_cars_profit_proxy_krw", ascending=False)


def format_krw(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}원"


def pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def write_report(
    daily: pd.DataFrame,
    summary: pd.DataFrame,
    periods: pd.DataFrame,
    classes: pd.DataFrame,
    opportunity: pd.DataFrame,
    profit: pd.DataFrame,
    reservations: pd.DataFrame,
    pre_reservations: pd.DataFrame,
    vehicle_master: pd.DataFrame,
) -> None:
    total_fleet_hours = daily["fleet_count"].sum() * 24
    total_booked_hours = daily["booked_hours"].sum()
    total_revenue = daily["revenue_allocated_krw"].sum()
    stockout_days = int(daily["is_stockout"].sum())
    model_day_count = len(daily)
    overall_utilization = total_booked_hours / total_fleet_hours
    stockout_model_day_rate = stockout_days / model_day_count
    extra_1_profit_total = opportunity["extra_1_cars_profit_proxy_krw"].sum()
    extra_2_profit_total = opportunity["extra_2_cars_profit_proxy_krw"].sum()
    top_stockout_name = summary.iloc[0]["model_name"] if not summary.empty else "-"
    top_opportunity_name = opportunity.iloc[0]["model_name"] if not opportunity.empty else "-"
    top_profit_name = profit.iloc[0]["model_name"] if not profit.empty else "-"

    def bar_chart(
        df: pd.DataFrame,
        label_col: str,
        value_col: str,
        formatter,
        max_value: float | None = None,
    ) -> str:
        rows = []
        values = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
        chart_max = max_value or values.max()
        if chart_max <= 0:
            chart_max = 1
        for _, row in df.iterrows():
            label = html_lib.escape(str(row[label_col]))
            value = float(row[value_col] or 0)
            width = max(1, min(100, value / chart_max * 100))
            rows.append(
                f'<div class="bar-row">'
                f'<div class="bar-label">{label}</div>'
                f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
                f'<div class="bar-value">{formatter(value)}</div>'
                f'</div>'
            )
        return "\n".join(rows)

    stockout_chart = bar_chart(
        summary.head(8),
        "model_name",
        "stockout_days",
        lambda v: f"{v:,.0f}일",
    )
    stockout_rate_chart = bar_chart(
        summary.sort_values("stockout_day_rate", ascending=False).head(8),
        "model_name",
        "stockout_day_rate",
        lambda v: f"{v * 100:.1f}%",
        max_value=1.0,
    )
    model_utilization_chart = bar_chart(
        summary.sort_values("utilization_rate", ascending=False).head(8),
        "model_name",
        "utilization_rate",
        lambda v: f"{v * 100:.1f}%",
        max_value=1.0,
    )
    opportunity_chart = bar_chart(
        opportunity.head(8),
        "model_name",
        "extra_1_cars_profit_proxy_krw",
        format_krw,
    )
    class_utilization_chart = bar_chart(
        classes.sort_values("utilization_rate", ascending=False).head(8),
        "vehicle_class",
        "utilization_rate",
        lambda v: f"{v * 100:.1f}%",
        max_value=0.4,
    )
    class_stockout_chart = bar_chart(
        classes.sort_values("stockout_days", ascending=False).head(8),
        "vehicle_class",
        "stockout_days",
        lambda v: f"{v:,.0f}일",
    )
    profit_chart = bar_chart(
        profit.head(8),
        "model_name",
        "profit_proxy_before_depreciation_krw",
        format_krw,
    )

    top_stockout = summary.head(15).copy()
    top_stockout["utilization_rate"] = top_stockout["utilization_rate"].map(pct)
    top_stockout["stockout_day_rate"] = top_stockout["stockout_day_rate"].map(pct)
    top_stockout["revenue_krw"] = top_stockout["revenue_krw"].map(format_krw)

    top_opp = opportunity.head(15).copy()
    money_cols = [
        "extra_1_cars_revenue_proxy_krw",
        "extra_1_cars_repair_cost_proxy_krw",
        "extra_1_cars_profit_proxy_krw",
        "extra_2_cars_revenue_proxy_krw",
        "extra_2_cars_repair_cost_proxy_krw",
        "extra_2_cars_profit_proxy_krw",
    ]
    for col in money_cols:
        top_opp[col] = top_opp[col].map(format_krw)
    top_profit = profit.head(15).copy()
    for col in [
        "booked_revenue_krw",
        "repair_cost_winsorized_krw",
        "repair_cost_removed_as_excess_krw",
        "profit_proxy_before_depreciation_krw",
    ]:
        top_profit[col] = top_profit[col].map(format_krw)
    top_profit["utilization_rate"] = top_profit["utilization_rate"].map(pct)
    top_profit["profit_margin_proxy_before_depreciation"] = top_profit[
        "profit_margin_proxy_before_depreciation"
    ].map(pct)

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>제주로 렌터카 품절/가동률/증차 기회 분석</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ font-size: 26px; margin-bottom: 6px; }}
    h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #d9e2ec; padding-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .label {{ color: #52606d; font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 650; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 10px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ background: #edf2f7; }}
    .note {{ color: #52606d; font-size: 13px; line-height: 1.5; }}
    .insights {{ margin: 10px 0 0 18px; color: #334e68; font-size: 13px; line-height: 1.6; }}
    .dash {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }}
    .panel {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #ffffff; }}
    .panel-title {{ font-size: 13px; font-weight: 650; margin-bottom: 10px; color: #243b53; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(120px, 1.25fr) minmax(140px, 2fr) 100px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-label {{ font-size: 12px; color: #334e68; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 10px; background: #edf2f7; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #2f80ed; border-radius: 5px; }}
    .bar-value {{ font-size: 12px; color: #243b53; text-align: right; }}
    @media (max-width: 900px) {{
      .grid, .dash {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <h1>제주로 렌터카 품절/가동률/증차 기회 분석</h1>
  <div class="note">분석 기간: {ANALYSIS_START.date()} ~ {ANALYSIS_END.date()}.
  차량 보유기간은 데이터상 실제 매입/매각일이 없어 차량 등록일~ERP soft-delete일을 사용한 프록시입니다.</div>
  <div class="grid">
    <div class="metric"><div class="label">유효 예약</div><div class="value">{len(reservations):,}건</div></div>
    <div class="metric"><div class="label">평균 가동률</div><div class="value">{pct(total_booked_hours / total_fleet_hours)}</div></div>
    <div class="metric"><div class="label">품절 모델-일자</div><div class="value">{stockout_days:,}일</div></div>
    <div class="metric"><div class="label">매출 프록시</div><div class="value">{format_krw(total_revenue)}</div></div>
  </div>
  <h2>결과 해석</h2>
  <ul class="insights">
    <li>전체 평균 가동률은 {pct(overall_utilization)}로 높지 않지만, 품절은 특정 모델과 차급에 집중되어 있습니다. 전체 보유 재고를 늘리기보다 품절 빈도가 높은 모델을 선별하는 방식이 더 적합합니다.</li>
    <li>품절 모델-일자는 {stockout_days:,}일로 전체 모델-일자의 {pct(stockout_model_day_rate)}입니다. 가장 품절일이 많은 모델은 {html_lib.escape(str(top_stockout_name))}입니다.</li>
    <li>추가 1대 기준 이익 프록시는 {format_krw(extra_1_profit_total)}, 추가 2대 기준은 {format_krw(extra_2_profit_total)}입니다. 증차 후보는 {html_lib.escape(str(top_opportunity_name))}처럼 품절일과 미전환 수요가 동시에 높은 모델부터 검토하는 것이 좋습니다.</li>
    <li>모델별 운영이익 프록시는 {html_lib.escape(str(top_profit_name))}이 가장 높습니다. 이 값은 요금 매출에서 과도 수리비를 보정한 수리비만 차감한 값이며, 감가상각은 데이터 부재로 아직 빠져 있습니다.</li>
    <li>다만 매입가/매각가가 없어 감가상각을 반영하지 못했으므로, 최종 구매 의사결정 전에는 차량별 실제 취득가와 예상 매각가를 넣어 재계산해야 합니다.</li>
  </ul>
  <h2>대시보드</h2>
  <div class="dash">
    <div class="panel"><div class="panel-title">품절일 상위 모델</div>{stockout_chart}</div>
    <div class="panel"><div class="panel-title">기간 내 품절율 상위 모델</div>{stockout_rate_chart}</div>
    <div class="panel"><div class="panel-title">평균 가동률 상위 모델</div>{model_utilization_chart}</div>
    <div class="panel"><div class="panel-title">1대 증차 이익 프록시 상위 모델</div>{opportunity_chart}</div>
    <div class="panel"><div class="panel-title">모델별 운영이익 프록시</div>{profit_chart}</div>
    <div class="panel"><div class="panel-title">차급별 가동률</div>{class_utilization_chart}</div>
    <div class="panel"><div class="panel-title">차급별 품절 모델-일자</div>{class_stockout_chart}</div>
  </div>
  <h2>계산 정의</h2>
  <p class="note">유효 예약은 취소를 제외한 RETURNED, CONFIRMED, PENDING, IN_USE, IN_PROGRESS입니다.
  품절일은 모델-일자별 최대 동시 예약 수가 해당 일자의 보유 차량 수 이상인 날입니다.
  가동률은 예약 시간 합계 / 보유 차량 시간입니다. 증차 이익은 미전환 사전예약이 품절일에 존재한다는 전제의 프록시이며,
  감가상각은 매입/매각 데이터 부재로 제외했고 수리비는 차량별 수리비 합계의 상위 1%를 winsorize해 시간당 비용으로 반영했습니다.</p>
  <h2>분석 과정</h2>
  <p class="note">1) 예약 데이터에서 취소/삭제 건을 제외하고 예약 시작~반납 시간을 모델별로 정리했습니다.
  2) 차량 등록 데이터에서 모델별 보유 차량 수를 일자 단위로 펼쳤습니다.
  3) 모델-일자별 예약 시간과 최대 동시 예약 대수를 계산해 보유대수와 비교했습니다.
  4) 최대 동시 예약 대수가 보유대수 이상인 날을 품절일로 보고, 연속된 품절일을 품절 기간으로 묶었습니다.
  5) 미전환 사전예약 중 품절일과 겹치는 수요를 추가 차량 1대/2대가 흡수할 수 있는 시간으로 제한해 증차 매출과 비용 프록시를 계산했습니다.</p>
  <h2>1대 증차 이익 프록시 로직</h2>
  <p class="note">품절일에 발생한 미전환 사전예약을 잠재 수요로 보고, 차량 1대가 하루에 추가로 흡수할 수 있는 시간은 최대 24시간으로 제한했습니다.
  모델별 기존 예약의 시간당 매출을 계산한 뒤, 흡수 가능 시간에 곱해 추가 매출을 추정했습니다.
  비용은 차량별 수리비 합계에서 상위 1% 과도값을 winsorize한 뒤 모델별 시간당 수리비로 환산해 차감했습니다.
  실제 매입가와 매각가가 없어 감가상각 비용은 이 프록시에는 포함하지 않았습니다.</p>
  <h2>품절 상위 모델</h2>
  {top_stockout[["model_name", "vehicle_class", "observed_days", "stockout_days", "stockout_day_rate", "stockout_period_count", "avg_stockout_period_days", "max_stockout_period_days", "utilization_rate", "revenue_krw"]].to_html(index=False, escape=False)}
  <h2>증차 이익 프록시 상위 모델</h2>
  {top_opp[["model_name", "vehicle_class", "latent_hours", "extra_1_cars_profit_proxy_krw", "extra_2_cars_profit_proxy_krw", "extra_1_cars_revenue_proxy_krw", "extra_2_cars_revenue_proxy_krw"]].to_html(index=False, escape=False)}
  <h2>모델별 운영이익 프록시</h2>
  {top_profit[["model_name", "vehicle_class", "fleet_vehicle_days", "booked_vehicle_days", "utilization_rate", "booked_revenue_krw", "repair_cost_winsorized_krw", "repair_cost_removed_as_excess_krw", "profit_proxy_before_depreciation_krw", "profit_margin_proxy_before_depreciation", "depreciation_status"]].to_html(index=False, escape=False)}
  <h2>차급 요약</h2>
  {classes.assign(utilization_rate=classes["utilization_rate"].map(pct), revenue_krw=classes["revenue_krw"].map(format_krw)).to_html(index=False, escape=False)}
  <h2>데이터 한계</h2>
  <p class="note">차량 데이터셋의 README와 missing_data_requests에 따르면 실제 취득일, 매입가, 매각일, 매각가는 제공되지 않았습니다.
  따라서 “보유기간”, “감가상각 비용”, “순이익”은 회계 확정값이 아니라 운영 프록시입니다.
  현재 차량 상태는 스냅샷 상태이므로 과거 일자별 사용 불가 차량을 정확히 제거하지 못합니다.</p>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    notes = [
        "# 제주로 렌터카 품절/가동률/증차 기회 분석",
        "",
        f"- 분석 기간: {ANALYSIS_START.date()} ~ {ANALYSIS_END.date()}",
        f"- 유효 예약: {len(reservations):,}건",
        f"- 평균 가동률: {pct(total_booked_hours / total_fleet_hours)}",
        f"- 품절 모델-일자: {stockout_days:,} / {model_day_count:,}",
        f"- 매출 프록시: {format_krw(total_revenue)}",
        "",
        "## 핵심 한계",
        "- 실제 취득일, 매입가, 매각일, 매각가가 제공되지 않아 감가상각 기반 순이익은 계산하지 않았습니다.",
        "- 보유기간은 차량 등록일~ERP soft-delete일 프록시입니다.",
        "- 증차 이익은 품절일에 미전환 사전예약이 존재한다는 전제의 운영 프록시입니다.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    reservations_raw, pre_raw, vehicle_master, maintenance = load_inputs()
    reservations = clean_reservations(reservations_raw)
    pre_reservations = clean_pre_reservations(pre_raw)
    daily_fleet = build_daily_fleet(vehicle_master)
    daily_booked = build_daily_usage(reservations, "final_price_krw")

    daily = daily_fleet.merge(
        daily_booked,
        on=["date", "model_name", "vehicle_class"],
        how="left",
    )
    for col in ["booked_hours", "revenue_allocated_krw", "peak_concurrent"]:
        daily[col] = pd.to_numeric(daily[col], errors="coerce").fillna(0)
    daily["fleet_hours"] = daily["fleet_count"] * 24
    daily["utilization_rate"] = np.where(
        daily["fleet_hours"] > 0, daily["booked_hours"] / daily["fleet_hours"], np.nan
    )
    daily["is_stockout"] = (daily["fleet_count"] > 0) & (
        daily["peak_concurrent"] >= daily["fleet_count"]
    )
    daily["over_capacity_peak"] = np.maximum(daily["peak_concurrent"] - daily["fleet_count"], 0)

    periods = consecutive_periods(daily)
    summary = model_summary(daily, periods)
    classes = class_summary(summary)
    opportunity = incremental_opportunity(daily, pre_reservations, maintenance)
    profit = model_profit_comparison(daily, maintenance)

    daily.to_csv(OUT_DIR / "daily_model_capacity.csv", index=False, encoding="utf-8-sig")
    periods.to_csv(OUT_DIR / "stockout_periods.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "model_capacity_summary.csv", index=False, encoding="utf-8-sig")
    classes.to_csv(OUT_DIR / "class_capacity_summary.csv", index=False, encoding="utf-8-sig")
    opportunity.to_csv(OUT_DIR / "incremental_vehicle_profit_proxy.csv", index=False, encoding="utf-8-sig")
    profit.to_csv(OUT_DIR / "model_profit_proxy.csv", index=False, encoding="utf-8-sig")
    write_report(
        daily=daily,
        summary=summary,
        periods=periods,
        classes=classes,
        opportunity=opportunity,
        profit=profit,
        reservations=reservations,
        pre_reservations=pre_reservations,
        vehicle_master=vehicle_master,
    )

    print(
        {
            "valid_reservations": len(reservations),
            "unbooked_pre_reservations": len(pre_reservations),
            "daily_rows": len(daily),
            "stockout_model_days": int(daily["is_stockout"].sum()),
            "overall_utilization": float(daily["booked_hours"].sum() / daily["fleet_hours"].sum()),
            "revenue_proxy_krw": float(daily["revenue_allocated_krw"].sum()),
        }
    )


if __name__ == "__main__":
    main()
