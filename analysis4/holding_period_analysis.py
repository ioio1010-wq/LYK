from __future__ import annotations

import html as html_lib
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RES_DIR = ROOT / "LYK_Jejuro_data_package_2026-08-11" / "data"
VEH_DIR = ROOT / "LYK_Jejuro_vehicle_dataset_CSV_2026-08-18"
OUT_DIR = ROOT / "analysis4"

ANALYSIS_START = pd.Timestamp("2025-01-01")
ANALYSIS_END = pd.Timestamp("2026-08-11")
VALID_STATUSES = {"RETURNED", "CONFIRMED", "PENDING", "IN_USE", "IN_PROGRESS"}
OLD_AGE_THRESHOLD = 6.0
AGE_BUCKET_ORDER = ["0-1년", "2-3년", "4-5년", "6-7년", "8년+"]


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False, **kwargs)


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.tz_localize(None)


def parse_model_years(model_name: object) -> tuple[float, float]:
    if not isinstance(model_name, str):
        return np.nan, np.nan
    years = [int(value) for value in re.findall(r"(\d{2})\s*년", model_name)]
    range_match = re.match(r"\s*(\d{2})\s*/\s*(\d{2})\s*년", model_name)
    if range_match:
        years.extend([int(range_match.group(1)), int(range_match.group(2))])
    if not years:
        return np.nan, np.nan
    full_years = [2000 + year if year <= 40 else 1900 + year for year in years]
    return float(min(full_years)), float(max(full_years))


def normalize_model_family(model_name: object) -> str:
    if not isinstance(model_name, str):
        return "미상"
    name = re.sub(r"^\s*\d{2}(?:\s*/\s*\d{2})?\s*년\s*", "", model_name)
    name = re.sub(r"\([^)]*\)", "", name)
    name = re.sub(r"\[[^]]*\]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    tokens = [
        "아반떼",
        "쏘나타",
        "K3",
        "K5",
        "레이",
        "모닝",
        "캐스퍼",
        "투싼",
        "스포티지",
        "쏘렌토",
        "싼타페",
        "팰리세이드",
        "코나",
        "카니발",
        "스타리아",
        "쏠라티",
        "G80",
        "GV70",
        "GV80",
        "EV3",
        "EV6",
        "아이오닉",
        "미니",
        "BMW 420i",
        "벤츠",
        "포르쉐",
        "머스탱",
        "랭글러",
    ]
    for token in tokens:
        if token.lower() in name.lower():
            return token
    return name


def vehicle_age_years(date: pd.Series, model_year: pd.Series) -> pd.Series:
    start = pd.to_datetime(model_year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    age = (date - start).dt.days / 365.25
    return age.clip(lower=0)


def age_bucket(series: pd.Series) -> pd.Series:
    return pd.cut(
        series,
        bins=[-0.01, 1.999, 3.999, 5.999, 7.999, 100],
        labels=AGE_BUCKET_ORDER,
    ).astype("object")


def date_range_half_open(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    if pd.isna(start) or pd.isna(end) or end <= start:
        return []
    first = start.normalize()
    last = (end - pd.Timedelta(microseconds=1)).normalize()
    return list(pd.date_range(first, last, freq="D"))


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reservations = read_csv(RES_DIR / "reservations.csv")
    pre = read_csv(RES_DIR / "pre_reservations.csv")
    users = read_csv(RES_DIR / "users.csv")
    models = read_csv(RES_DIR / "vehicle_models.csv")
    vehicles = read_csv(VEH_DIR / "vehicle_master.csv")

    year_df = models["vehicle_model_name"].apply(parse_model_years).apply(pd.Series)
    models["model_year_min"] = year_df[0]
    models["model_year_latest"] = year_df[1]
    models["model_family"] = models["vehicle_model_name"].map(normalize_model_family)

    pre_link = (
        pre[pre["reservation_key"].notna()]
        .sort_values("created_at_utc")
        .drop_duplicates("reservation_key")
        [
            [
                "reservation_key",
                "user_key",
                "insurance_name_code",
                "is_full_coverage",
                "discount_krw",
                "loyalty_mileage_used",
            ]
        ]
    )

    reservations = reservations.merge(
        models[
            [
                "vehicle_model_key",
                "vehicle_model_name",
                "vehicle_model_title",
                "vehicle_class",
                "manufacturer",
                "fuel",
                "seat_count",
                "model_year_min",
                "model_year_latest",
                "model_family",
            ]
        ],
        on="vehicle_model_key",
        how="left",
    )
    reservations = reservations.merge(pre_link, on="reservation_key", how="left")
    reservations = reservations.merge(users, on="user_key", how="left", suffixes=("", "_user"))
    return reservations, vehicles, users


def clean_reservations(reservations: pd.DataFrame) -> pd.DataFrame:
    df = reservations.copy()
    df["pickup_at"] = parse_dt(df["pickup_at_source"])
    df["dropoff_at"] = parse_dt(df["dropoff_at_source"])
    df["final_price_krw"] = pd.to_numeric(df["final_price_krw"], errors="coerce").fillna(0)
    df["rental_duration_hours"] = pd.to_numeric(df["rental_duration_hours"], errors="coerce")
    df["discount_krw"] = pd.to_numeric(df["discount_krw"], errors="coerce").fillna(0)
    df["loyalty_mileage_used"] = pd.to_numeric(df["loyalty_mileage_used"], errors="coerce").fillna(0)
    mask = (
        df["reservation_status"].isin(VALID_STATUSES)
        & ~df["is_deleted"].astype(bool)
        & df["pickup_at"].notna()
        & df["dropoff_at"].notna()
        & (df["dropoff_at"] > df["pickup_at"])
        & (df["pickup_at"] >= ANALYSIS_START)
        & (df["pickup_at"] <= ANALYSIS_END)
        & df["model_year_latest"].notna()
    )
    df = df.loc[mask].copy()
    df["age_at_pickup_years"] = vehicle_age_years(df["pickup_at"], df["model_year_latest"])
    df["age_bucket"] = age_bucket(df["age_at_pickup_years"])
    df["pickup_month"] = df["pickup_at"].dt.to_period("M").astype(str)
    df["has_joined_user"] = df["user_key"].notna()
    df["is_repeat_user"] = pd.to_numeric(
        df["linked_reservation_count"], errors="coerce"
    ).fillna(0) >= 2
    df["is_returned_repeat_user"] = pd.to_numeric(
        df["returned_reservation_count"], errors="coerce"
    ).fillna(0) >= 2
    spend = pd.to_numeric(df["sum_linked_reservation_final_price_krw"], errors="coerce")
    high_value_cutoff = spend.dropna().quantile(0.75) if spend.notna().any() else np.nan
    df["is_high_value_user"] = spend >= high_value_cutoff
    return df


def build_daily_fleet(vehicles: pd.DataFrame) -> pd.DataFrame:
    df = vehicles.copy()
    df["model_year_latest"] = df["model_name"].map(parse_model_years).map(lambda item: item[1])
    df["model_family"] = df["model_name"].map(normalize_model_family)
    df["fleet_start"] = pd.to_datetime(df["vehicle_registration_date_observed"], errors="coerce")
    df["fleet_start"] = df["fleet_start"].fillna(parse_dt(df["erp_record_created_at_utc"]))
    df["fleet_end"] = parse_dt(df["erp_soft_deleted_at_utc"]).fillna(ANALYSIS_END + pd.Timedelta(days=1))
    df["fleet_start"] = df["fleet_start"].clip(lower=ANALYSIS_START)
    df["fleet_end"] = df["fleet_end"].clip(upper=ANALYSIS_END + pd.Timedelta(days=1))
    df = df[
        df["model_name"].notna()
        & df["model_year_latest"].notna()
        & df["fleet_start"].notna()
        & df["fleet_end"].notna()
        & (df["fleet_end"] > df["fleet_start"])
    ].copy()
    rows: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        for day in date_range_half_open(row.fleet_start, row.fleet_end):
            rows.append(
                {
                    "date": day,
                    "model_name": row.model_name,
                    "vehicle_class": row.vehicle_class,
                    "model_year_latest": row.model_year_latest,
                    "model_family": row.model_family,
                    "vehicle_id": row.vehicle_id,
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    daily = (
        daily.groupby(
            ["date", "model_name", "vehicle_class", "model_year_latest", "model_family"],
            as_index=False,
        )
        .agg(fleet_count=("vehicle_id", "nunique"))
    )
    daily["vehicle_age_years"] = vehicle_age_years(daily["date"], daily["model_year_latest"])
    daily["age_bucket"] = age_bucket(daily["vehicle_age_years"])
    return daily


def build_daily_revenue(reservations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in reservations.itertuples(index=False):
        start = row.pickup_at
        end = row.dropoff_at
        total_hours = max((end - start).total_seconds() / 3600, 0)
        if total_hours <= 0:
            continue
        revenue_per_hour = row.final_price_krw / total_hours
        for day in date_range_half_open(start, min(end, ANALYSIS_END + pd.Timedelta(days=1))):
            overlap_start = max(start, day)
            overlap_end = min(end, day + pd.Timedelta(days=1), ANALYSIS_END + pd.Timedelta(days=1))
            hours = max((overlap_end - overlap_start).total_seconds() / 3600, 0)
            if hours <= 0:
                continue
            age = max((day - pd.Timestamp(int(row.model_year_latest), 1, 1)).days / 365.25, 0)
            rows.append(
                {
                    "date": day,
                    "model_name": row.vehicle_model_name,
                    "vehicle_class": row.vehicle_class,
                    "model_year_latest": row.model_year_latest,
                    "model_family": row.model_family,
                    "booked_hours": hours,
                    "revenue_krw": revenue_per_hour * hours,
                    "vehicle_age_years": age,
                }
            )
    daily = pd.DataFrame(rows)
    if daily.empty:
        return pd.DataFrame()
    daily["age_bucket"] = age_bucket(daily["vehicle_age_years"])
    return (
        daily.groupby(
            [
                "date",
                "model_name",
                "vehicle_class",
                "model_year_latest",
                "model_family",
                "age_bucket",
            ],
            as_index=False,
        )
        .agg(booked_hours=("booked_hours", "sum"), revenue_krw=("revenue_krw", "sum"))
    )


def combine_daily(fleet: pd.DataFrame, booked: pd.DataFrame) -> pd.DataFrame:
    daily = fleet.merge(
        booked,
        on=[
            "date",
            "model_name",
            "vehicle_class",
            "model_year_latest",
            "model_family",
            "age_bucket",
        ],
        how="left",
    )
    daily["booked_hours"] = daily["booked_hours"].fillna(0)
    daily["revenue_krw"] = daily["revenue_krw"].fillna(0)
    daily["fleet_vehicle_days"] = daily["fleet_count"]
    daily["booked_vehicle_days"] = daily["booked_hours"] / 24
    daily["utilization_rate"] = np.where(
        daily["fleet_vehicle_days"] > 0,
        daily["booked_vehicle_days"] / daily["fleet_vehicle_days"],
        np.nan,
    )
    daily["revenue_per_fleet_day"] = np.where(
        daily["fleet_vehicle_days"] > 0,
        daily["revenue_krw"] / daily["fleet_vehicle_days"],
        np.nan,
    )
    return daily


def summarize_age_curves(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    class_age = (
        daily.groupby(["vehicle_class", "age_bucket"], as_index=False, observed=False)
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
            active_model_days=("date", "nunique"),
        )
    )
    class_age["utilization_rate"] = np.where(
        class_age["fleet_vehicle_days"] > 0,
        class_age["booked_vehicle_days"] / class_age["fleet_vehicle_days"],
        np.nan,
    )
    class_age["revenue_per_fleet_day"] = np.where(
        class_age["fleet_vehicle_days"] > 0,
        class_age["revenue_krw"] / class_age["fleet_vehicle_days"],
        np.nan,
    )

    total_age = (
        daily.groupby("age_bucket", as_index=False, observed=False)
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
        )
    )
    total_age["utilization_rate"] = np.where(
        total_age["fleet_vehicle_days"] > 0,
        total_age["booked_vehicle_days"] / total_age["fleet_vehicle_days"],
        np.nan,
    )
    total_age["revenue_per_fleet_day"] = np.where(
        total_age["fleet_vehicle_days"] > 0,
        total_age["revenue_krw"] / total_age["fleet_vehicle_days"],
        np.nan,
    )
    return total_age, class_age


def customer_profile(reservations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = reservations.copy()
    df["age_segment"] = np.where(
        df["age_at_pickup_years"] >= OLD_AGE_THRESHOLD,
        "6년+ 오래된 연식",
        "0-5년 비교군",
    )
    spend = pd.to_numeric(df["sum_linked_reservation_final_price_krw"], errors="coerce")
    linked = df[df["has_joined_user"]].copy()
    profile = (
        linked.groupby("age_segment", as_index=False)
        .agg(
            joined_reservations=("reservation_key", "nunique"),
            unique_users=("user_key", "nunique"),
            repeat_user_share=("is_repeat_user", "mean"),
            returned_repeat_user_share=("is_returned_repeat_user", "mean"),
            high_value_user_share=("is_high_value_user", "mean"),
            marketing_agreed_share=("marketing_agreed_current", "mean"),
            avg_user_linked_reservations=("linked_reservation_count", "mean"),
            avg_user_total_spend_krw=("sum_linked_reservation_final_price_krw", "mean"),
            avg_rental_hours=("rental_duration_hours", "mean"),
            avg_reservation_price_krw=("final_price_krw", "mean"),
            full_coverage_share=("is_full_coverage", "mean"),
            avg_discount_krw=("discount_krw", "mean"),
            avg_loyalty_mileage_used=("loyalty_mileage_used", "mean"),
        )
    )
    profile["user_join_scope"] = "reservation_key joined to user_key only"

    class_profile = (
        linked[linked["age_at_pickup_years"] >= OLD_AGE_THRESHOLD]
        .groupby("vehicle_class", as_index=False)
        .agg(
            joined_reservations=("reservation_key", "nunique"),
            unique_users=("user_key", "nunique"),
            repeat_user_share=("is_repeat_user", "mean"),
            high_value_user_share=("is_high_value_user", "mean"),
            avg_reservation_price_krw=("final_price_krw", "mean"),
            avg_rental_hours=("rental_duration_hours", "mean"),
        )
        .sort_values("joined_reservations", ascending=False)
    )
    return profile, class_profile


def model_signal(daily: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    daily_model = (
        daily.groupby(
            ["model_name", "vehicle_class", "model_family", "model_year_latest"], as_index=False
        )
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
            avg_vehicle_age_years=("vehicle_age_years", "mean"),
            last_observed_age_years=("vehicle_age_years", "max"),
        )
    )
    daily_model["utilization_rate"] = np.where(
        daily_model["fleet_vehicle_days"] > 0,
        daily_model["booked_vehicle_days"] / daily_model["fleet_vehicle_days"],
        np.nan,
    )
    daily_model["revenue_per_fleet_day"] = np.where(
        daily_model["fleet_vehicle_days"] > 0,
        daily_model["revenue_krw"] / daily_model["fleet_vehicle_days"],
        np.nan,
    )

    old_daily = daily[daily["vehicle_age_years"] >= OLD_AGE_THRESHOLD]
    old_model = (
        old_daily.groupby(
            ["model_name", "vehicle_class", "model_family", "model_year_latest"], as_index=False
        )
        .agg(
            old_fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            old_booked_vehicle_days=("booked_vehicle_days", "sum"),
            old_revenue_krw=("revenue_krw", "sum"),
        )
    )
    old_model["old_utilization_rate"] = np.where(
        old_model["old_fleet_vehicle_days"] > 0,
        old_model["old_booked_vehicle_days"] / old_model["old_fleet_vehicle_days"],
        np.nan,
    )
    old_model["old_revenue_per_fleet_day"] = np.where(
        old_model["old_fleet_vehicle_days"] > 0,
        old_model["old_revenue_krw"] / old_model["old_fleet_vehicle_days"],
        np.nan,
    )

    res_model = (
        reservations.groupby(
            ["vehicle_model_name", "vehicle_class", "model_family", "model_year_latest"],
            as_index=False,
        )
        .agg(
            reservation_count=("reservation_key", "nunique"),
            unique_joined_users=("user_key", "nunique"),
            active_months=("pickup_month", "nunique"),
            avg_reservation_price_krw=("final_price_krw", "mean"),
            avg_rental_hours=("rental_duration_hours", "mean"),
        )
        .rename(columns={"vehicle_model_name": "model_name"})
    )
    old_res = reservations[reservations["age_at_pickup_years"] >= OLD_AGE_THRESHOLD]
    old_res_model = (
        old_res.groupby(
            ["vehicle_model_name", "vehicle_class", "model_family", "model_year_latest"],
            as_index=False,
        )
        .agg(
            old_reservation_count=("reservation_key", "nunique"),
            old_unique_joined_users=("user_key", "nunique"),
            old_active_months=("pickup_month", "nunique"),
            old_avg_reservation_price_krw=("final_price_krw", "mean"),
            old_repeat_user_share=("is_repeat_user", "mean"),
            old_high_value_user_share=("is_high_value_user", "mean"),
        )
        .rename(columns={"vehicle_model_name": "model_name"})
    )

    out = daily_model.merge(
        old_model,
        on=["model_name", "vehicle_class", "model_family", "model_year_latest"],
        how="left",
    )
    out = out.merge(
        res_model,
        on=["model_name", "vehicle_class", "model_family", "model_year_latest"],
        how="left",
    )
    out = out.merge(
        old_res_model,
        on=["model_name", "vehicle_class", "model_family", "model_year_latest"],
        how="left",
    )
    fill_zero = [
        "old_fleet_vehicle_days",
        "old_booked_vehicle_days",
        "old_revenue_krw",
        "old_reservation_count",
        "old_unique_joined_users",
        "old_active_months",
    ]
    out[fill_zero] = out[fill_zero].fillna(0)
    out["old_revenue_share"] = np.where(
        out["revenue_krw"] > 0, out["old_revenue_krw"] / out["revenue_krw"], np.nan
    )
    class_old_median = (
        out[out["old_fleet_vehicle_days"] > 0]
        .groupby("vehicle_class")["old_revenue_per_fleet_day"]
        .median()
        .rename("class_old_revenue_per_fleet_day_median")
        .reset_index()
    )
    out = out.merge(class_old_median, on="vehicle_class", how="left")
    out["old_revenue_vs_class_median"] = np.where(
        out["class_old_revenue_per_fleet_day_median"] > 0,
        out["old_revenue_per_fleet_day"] / out["class_old_revenue_per_fleet_day_median"],
        np.nan,
    )
    raw_score = (
        out["old_revenue_vs_class_median"].fillna(0).clip(upper=3) * 35
        + out["old_utilization_rate"].fillna(0).clip(upper=0.5) / 0.5 * 30
        + (out["old_active_months"].fillna(0).clip(upper=12) / 12) * 20
        + np.log1p(out["old_reservation_count"].fillna(0)).clip(upper=np.log1p(150))
        / np.log1p(150)
        * 15
    )
    out["long_hold_score"] = raw_score / 170 * 100
    conditions = [
        (out["old_reservation_count"] >= 50)
        & (out["old_active_months"] >= 6)
        & (out["old_revenue_vs_class_median"] >= 1.0),
        (out["last_observed_age_years"] >= OLD_AGE_THRESHOLD)
        & (
            (out["old_reservation_count"] < 10)
            | (out["old_revenue_vs_class_median"] < 0.5)
        ),
    ]
    choices = ["길게 보유 검토", "교체/매각 검토"]
    out["holding_period_signal"] = np.select(conditions, choices, default="추가 관찰")
    out["recommended_holding_period_proxy"] = np.select(
        [
            out["holding_period_signal"].eq("길게 보유 검토"),
            out["holding_period_signal"].eq("교체/매각 검토")
            & (out["last_observed_age_years"] >= OLD_AGE_THRESHOLD),
            out["last_observed_age_years"] < OLD_AGE_THRESHOLD,
        ],
        ["6년+ 장기 보유 검증", "6년 전후 교체 검토", "6년 도달 전 추가 관찰"],
        default="4-6년 구간 모니터링",
    )
    return out.sort_values("long_hold_score", ascending=False)


def family_year_comparison(daily: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    fam = (
        daily.groupby(["model_family", "vehicle_class", "model_year_latest"], as_index=False)
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
            avg_age_years=("vehicle_age_years", "mean"),
        )
    )
    fam["utilization_rate"] = np.where(
        fam["fleet_vehicle_days"] > 0,
        fam["booked_vehicle_days"] / fam["fleet_vehicle_days"],
        np.nan,
    )
    fam["revenue_per_fleet_day"] = np.where(
        fam["fleet_vehicle_days"] > 0,
        fam["revenue_krw"] / fam["fleet_vehicle_days"],
        np.nan,
    )
    res = (
        reservations.groupby(["model_family", "vehicle_class", "model_year_latest"], as_index=False)
        .agg(
            reservation_count=("reservation_key", "nunique"),
            unique_joined_users=("user_key", "nunique"),
            avg_reservation_price_krw=("final_price_krw", "mean"),
        )
    )
    fam = fam.merge(res, on=["model_family", "vehicle_class", "model_year_latest"], how="left")
    fam["family_year_rank_by_revenue_per_fleet_day"] = fam.groupby("model_family")[
        "revenue_per_fleet_day"
    ].rank(ascending=False, method="dense")
    return fam.sort_values(["model_family", "model_year_latest"])


def class_holding_recommendation(class_age: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    order = {bucket: idx for idx, bucket in enumerate(AGE_BUCKET_ORDER)}
    for vehicle_class, group in class_age.groupby("vehicle_class"):
        group = group.copy()
        group["bucket_order"] = group["age_bucket"].map(order)
        group = group.sort_values("bucket_order")
        baseline = group.loc[group["age_bucket"] == "0-1년", "revenue_per_fleet_day"]
        baseline_util = group.loc[group["age_bucket"] == "0-1년", "utilization_rate"]
        baseline_value = float(baseline.iloc[0]) if len(baseline) and baseline.iloc[0] > 0 else np.nan
        baseline_util_value = (
            float(baseline_util.iloc[0])
            if len(baseline_util) and baseline_util.iloc[0] > 0
            else np.nan
        )
        group["revenue_retention_vs_0_1"] = np.where(
            baseline_value > 0,
            group["revenue_per_fleet_day"] / baseline_value,
            np.nan,
        )
        group["utilization_retention_vs_0_1"] = np.where(
            baseline_util_value > 0,
            group["utilization_rate"] / baseline_util_value,
            np.nan,
        )
        drop = group[
            (group["revenue_retention_vs_0_1"] < 0.5)
            & (group["utilization_retention_vs_0_1"] < 0.5)
            & (group["fleet_vehicle_days"] >= 100)
        ]
        if drop.empty:
            review_bucket = "8년+까지 추가 관찰"
            reason = "신차급 대비 매출/가동률이 동시에 50% 미만으로 하락한 충분한 구간 없음"
        else:
            review_bucket = str(drop.iloc[0]["age_bucket"])
            reason = "보유일당 매출과 가동률이 신차급 대비 50% 미만으로 동시 하락"
        old = group[group["age_bucket"].isin(["6-7년", "8년+"])]
        rows.append(
            {
                "vehicle_class": vehicle_class,
                "baseline_0_1_revenue_per_fleet_day": baseline_value,
                "baseline_0_1_utilization_rate": baseline_util_value,
                "replacement_review_start_age_bucket": review_bucket,
                "old_age_revenue_per_fleet_day": old["revenue_per_fleet_day"].mean(),
                "old_age_utilization_rate": old["utilization_rate"].mean(),
                "recommendation_reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values("replacement_review_start_age_bucket")


def format_krw(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}원"


def pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def number(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def bar_chart(df: pd.DataFrame, label_col: str, value_col: str, formatter, max_value=None) -> str:
    if df.empty:
        return '<div class="note">표시할 데이터가 없습니다.</div>'
    chart_max = max_value or pd.to_numeric(df[value_col], errors="coerce").fillna(0).max()
    chart_max = chart_max if chart_max and chart_max > 0 else 1
    rows = []
    for _, row in df.iterrows():
        label = html_lib.escape(str(row[label_col]))
        value = float(row[value_col] or 0)
        width = max(1, min(100, value / chart_max * 100))
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
            f'<div class="bar-value">{formatter(value)}</div></div>'
        )
    return "\n".join(rows)


def write_report(
    reservations: pd.DataFrame,
    daily: pd.DataFrame,
    total_age: pd.DataFrame,
    class_age: pd.DataFrame,
    class_recommendations: pd.DataFrame,
    model_signals: pd.DataFrame,
    family_year: pd.DataFrame,
    customer_profiles: pd.DataFrame,
    customer_class_profiles: pd.DataFrame,
) -> None:
    old_res = reservations[reservations["age_at_pickup_years"] >= OLD_AGE_THRESHOLD]
    old_revenue = old_res["final_price_krw"].sum()
    total_revenue = reservations["final_price_krw"].sum()
    joined_rate = reservations["has_joined_user"].mean()
    long_candidates = model_signals[model_signals["holding_period_signal"] == "길게 보유 검토"]
    replacement_candidates = model_signals[
        model_signals["holding_period_signal"] == "교체/매각 검토"
    ]
    top_long = long_candidates.iloc[0]["model_name"] if not long_candidates.empty else "-"
    top_age_bucket = (
        total_age.sort_values("revenue_per_fleet_day", ascending=False).iloc[0]["age_bucket"]
        if not total_age.empty
        else "-"
    )

    age_chart = bar_chart(
        total_age.sort_values("age_bucket", key=lambda s: s.map({v: i for i, v in enumerate(AGE_BUCKET_ORDER)})),
        "age_bucket",
        "revenue_per_fleet_day",
        format_krw,
    )
    old_model_chart = bar_chart(
        model_signals[model_signals["old_reservation_count"] > 0].head(8),
        "model_name",
        "old_revenue_krw",
        format_krw,
    )
    long_score_chart = bar_chart(
        model_signals.head(8),
        "model_name",
        "long_hold_score",
        lambda v: f"{v:.0f}점",
    )
    class_old_chart = bar_chart(
        class_age[class_age["age_bucket"].isin(["6-7년", "8년+"])]
        .groupby("vehicle_class", as_index=False)
        .agg(revenue_per_fleet_day=("revenue_per_fleet_day", "mean"))
        .sort_values("revenue_per_fleet_day", ascending=False)
        .head(8),
        "vehicle_class",
        "revenue_per_fleet_day",
        format_krw,
    )

    display_signals = model_signals.head(20).copy()
    for col in ["revenue_krw", "old_revenue_krw", "revenue_per_fleet_day", "old_revenue_per_fleet_day"]:
        display_signals[col] = display_signals[col].map(format_krw)
    for col in ["utilization_rate", "old_utilization_rate", "old_revenue_share"]:
        display_signals[col] = display_signals[col].map(pct)
    display_signals["long_hold_score"] = display_signals["long_hold_score"].map(lambda v: f"{v:.0f}")

    display_class_age = class_age.copy()
    display_class_age["revenue_krw"] = display_class_age["revenue_krw"].map(format_krw)
    display_class_age["revenue_per_fleet_day"] = display_class_age["revenue_per_fleet_day"].map(format_krw)
    display_class_age["utilization_rate"] = display_class_age["utilization_rate"].map(pct)

    display_class_recommendations = class_recommendations.copy()
    for col in ["baseline_0_1_revenue_per_fleet_day", "old_age_revenue_per_fleet_day"]:
        display_class_recommendations[col] = display_class_recommendations[col].map(format_krw)
    for col in ["baseline_0_1_utilization_rate", "old_age_utilization_rate"]:
        display_class_recommendations[col] = display_class_recommendations[col].map(pct)

    display_family = (
        family_year.sort_values("revenue_per_fleet_day", ascending=False)
        .head(30)
        .copy()
    )
    for col in ["revenue_krw", "revenue_per_fleet_day", "avg_reservation_price_krw"]:
        display_family[col] = display_family[col].map(format_krw)
    display_family["utilization_rate"] = display_family["utilization_rate"].map(pct)

    display_customer = customer_profiles.copy()
    for col in [
        "repeat_user_share",
        "returned_repeat_user_share",
        "high_value_user_share",
        "marketing_agreed_share",
        "full_coverage_share",
    ]:
        display_customer[col] = display_customer[col].map(pct)
    for col in ["avg_user_total_spend_krw", "avg_reservation_price_krw", "avg_discount_krw"]:
        display_customer[col] = display_customer[col].map(format_krw)

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>제주로 렌터카 적정 보유기간 신호 분석</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ font-size: 26px; margin-bottom: 6px; }}
    h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #d9e2ec; padding-bottom: 6px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .label {{ color: #52606d; font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 650; margin-top: 4px; }}
    .note {{ color: #52606d; font-size: 13px; line-height: 1.5; }}
    .insights {{ margin: 10px 0 0 18px; color: #334e68; font-size: 13px; line-height: 1.6; }}
    .dash {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }}
    .panel {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #ffffff; }}
    .panel-title {{ font-size: 13px; font-weight: 650; margin-bottom: 10px; color: #243b53; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(120px, 1.25fr) minmax(140px, 2fr) 110px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-label {{ font-size: 12px; color: #334e68; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 10px; background: #edf2f7; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: #2f80ed; border-radius: 5px; }}
    .bar-value {{ font-size: 12px; color: #243b53; text-align: right; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 10px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #edf2f7; }}
    @media (max-width: 900px) {{
      .grid, .dash {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <h1>제주로 렌터카 적정 보유기간 신호 분석</h1>
  <div class="note">분석 기간: {ANALYSIS_START.date()} ~ {ANALYSIS_END.date()}.
  실제 매입일·매입가·매각일·매각가가 없어, 연식과 차량 등록일 기반 프록시로 매출 유지력을 봤습니다.</div>
  <div class="grid">
    <div class="metric"><div class="label">유효 예약</div><div class="value">{len(reservations):,}건</div></div>
    <div class="metric"><div class="label">6년+ 연식 매출</div><div class="value">{format_krw(old_revenue)}</div></div>
    <div class="metric"><div class="label">6년+ 매출 비중</div><div class="value">{pct(old_revenue / total_revenue)}</div></div>
    <div class="metric"><div class="label">사용자 조인율</div><div class="value">{pct(joined_rate)}</div></div>
  </div>
  <h2>결과 해석</h2>
  <ul class="insights">
    <li>이 데이터만으로 감가상각 곡선과 매출 감소 속도를 직접 비교할 수는 없습니다. 매입가·매각가가 모두 비어 있어, 이번 결과는 보유기간 의사결정의 “수요/매출 유지 신호”로 보는 것이 맞습니다.</li>
    <li>보유일당 매출이 가장 높은 차령 구간은 {html_lib.escape(str(top_age_bucket))}입니다. 전체 평균만 보면 오래된 차량을 일괄 매각하기보다 차급과 차종별로 분리 판단해야 합니다.</li>
    <li>6년 이상 연식에서도 수요가 유지되는 대표 후보는 {html_lib.escape(str(top_long))}입니다. 이 그룹은 실제 감가 데이터가 들어오면 “장기 보유” 후보로 우선 검증할 가치가 있습니다.</li>
    <li>오래된 차량 예약 고객은 조인 가능한 사용자 기준으로 반복 예약 여부, 고가치 고객 비중, 평균 예약금액을 비교했습니다. 개인정보가 제거되어 성별·연령·지역 같은 고객 속성은 확인할 수 없습니다.</li>
  </ul>
  <h2>대시보드</h2>
  <div class="dash">
    <div class="panel"><div class="panel-title">차령 구간별 보유일당 매출</div>{age_chart}</div>
    <div class="panel"><div class="panel-title">6년+ 매출 상위 차종</div>{old_model_chart}</div>
    <div class="panel"><div class="panel-title">장기 보유 신호 점수 상위 차종</div>{long_score_chart}</div>
    <div class="panel"><div class="panel-title">6년+ 차급별 보유일당 매출</div>{class_old_chart}</div>
  </div>
  <h2>분석 과정</h2>
  <p class="note">1) 예약에서 취소/삭제 건을 제외하고 모델명, 연식, 차급, 매출을 붙였습니다.
  2) 모델명에서 연식을 추출해 예약 시점의 차령을 계산했습니다. 예: 20년 모델이 2026년에 예약되면 약 6년차로 봅니다.
  3) 차량 등록 데이터로 일자별 보유 차량 수를 펼치고, 예약 시간을 일자별 매출로 배분했습니다.
  4) 차령 구간별 보유일당 매출, 가동률, 예약 수를 계산했습니다.
  5) 6년 이상 연식에서도 예약·매출·가동률이 차급 중앙값 이상으로 유지되는 차종에 장기 보유 신호를 부여했습니다.
  6) 차급별로 보유일당 매출과 가동률이 신차급 대비 50% 미만으로 동시에 하락하는 첫 차령 구간을 교체 검토 시작 구간으로 표시했습니다.</p>
  <h2>적정 보유기간 신호</h2>
  {display_signals[["model_name", "vehicle_class", "model_year_latest", "last_observed_age_years", "reservation_count", "revenue_krw", "revenue_per_fleet_day", "utilization_rate", "old_reservation_count", "old_revenue_krw", "old_revenue_per_fleet_day", "old_utilization_rate", "old_revenue_share", "long_hold_score", "holding_period_signal", "recommended_holding_period_proxy"]].to_html(index=False, escape=False)}
  <h2>차급별 교체 검토 구간</h2>
  {display_class_recommendations.to_html(index=False, escape=False)}
  <h2>차급별 차령 비교</h2>
  {display_class_age[["vehicle_class", "age_bucket", "fleet_vehicle_days", "booked_vehicle_days", "revenue_krw", "revenue_per_fleet_day", "utilization_rate"]].to_html(index=False, escape=False)}
  <h2>같은 차종 내 연식 비교</h2>
  {display_family[["model_family", "vehicle_class", "model_year_latest", "reservation_count", "revenue_krw", "revenue_per_fleet_day", "utilization_rate", "family_year_rank_by_revenue_per_fleet_day"]].to_html(index=False, escape=False)}
  <h2>오래된 차량 예약 고객 프로필</h2>
  {display_customer.to_html(index=False, escape=False)}
  <h2>데이터 한계</h2>
  <p class="note">차량 데이터셋의 취득일, 매입가, 매각일, 매각가는 모두 unavailable입니다.
  따라서 실제 적정 보유기간은 이번 리포트의 장기 보유 신호에 차량별 감가상각과 정비비 전망을 추가해야 확정할 수 있습니다.
  또한 예약은 차량 개별 ID가 아니라 모델 단위로 연결되어 있어, 대부분의 분석은 모델/차급 단위 프록시입니다.</p>
</body>
</html>
"""
    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    notes = [
        "# 제주로 렌터카 적정 보유기간 신호 분석",
        "",
        f"- 분석 기간: {ANALYSIS_START.date()} ~ {ANALYSIS_END.date()}",
        f"- 유효 예약: {len(reservations):,}건",
        f"- 6년+ 연식 매출: {format_krw(old_revenue)}",
        f"- 6년+ 매출 비중: {pct(old_revenue / total_revenue)}",
        f"- 사용자 조인율: {pct(joined_rate)}",
        "",
        "## 핵심 한계",
        "- 실제 매입일, 매입가, 매각일, 매각가가 없어 감가상각 기반 적정 보유기간은 계산하지 않았습니다.",
        "- 이번 결과는 차령별 매출 유지력과 고객 프로필을 기반으로 한 운영 신호입니다.",
    ]
    (OUT_DIR / "summary.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    reservations_raw, vehicles, _ = load_data()
    reservations = clean_reservations(reservations_raw)
    fleet = build_daily_fleet(vehicles)
    booked = build_daily_revenue(reservations)
    daily = combine_daily(fleet, booked)
    total_age, class_age = summarize_age_curves(daily)
    class_recommendations = class_holding_recommendation(class_age)
    profiles, class_profiles = customer_profile(reservations)
    signals = model_signal(daily, reservations)
    family_year = family_year_comparison(daily, reservations)

    daily.to_csv(OUT_DIR / "daily_age_revenue.csv", index=False, encoding="utf-8-sig")
    total_age.to_csv(OUT_DIR / "age_curve_summary.csv", index=False, encoding="utf-8-sig")
    class_age.to_csv(OUT_DIR / "class_age_summary.csv", index=False, encoding="utf-8-sig")
    class_recommendations.to_csv(
        OUT_DIR / "class_holding_period_recommendation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    signals.to_csv(OUT_DIR / "model_holding_period_signal.csv", index=False, encoding="utf-8-sig")
    family_year.to_csv(OUT_DIR / "family_year_comparison.csv", index=False, encoding="utf-8-sig")
    profiles.to_csv(OUT_DIR / "old_vehicle_customer_profile.csv", index=False, encoding="utf-8-sig")
    class_profiles.to_csv(
        OUT_DIR / "old_vehicle_customer_profile_by_class.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_report(
        reservations=reservations,
        daily=daily,
        total_age=total_age,
        class_age=class_age,
        class_recommendations=class_recommendations,
        model_signals=signals,
        family_year=family_year,
        customer_profiles=profiles,
        customer_class_profiles=class_profiles,
    )
    print(
        {
            "valid_reservations": len(reservations),
            "daily_rows": len(daily),
            "old_age_reservations": int(
                (reservations["age_at_pickup_years"] >= OLD_AGE_THRESHOLD).sum()
            ),
            "old_age_revenue_krw": float(
                reservations.loc[
                    reservations["age_at_pickup_years"] >= OLD_AGE_THRESHOLD,
                    "final_price_krw",
                ].sum()
            ),
            "long_hold_candidates": int((signals["holding_period_signal"] == "길게 보유 검토").sum()),
            "replacement_candidates": int((signals["holding_period_signal"] == "교체/매각 검토").sum()),
        }
    )


if __name__ == "__main__":
    main()
