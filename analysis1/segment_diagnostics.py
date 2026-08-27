from __future__ import annotations

import html
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path("/Users/yangjiyun/Desktop/LYK")
PROCESSED = BASE / "data" / "processed"
OUT = BASE / "analysis1" / "segment_diagnostics"

STAGE_ORDER = [
    "anonymous_visit",
    "identified_visit",
    "search_started",
    "car_clicked",
    "booking_started",
    "insurance_viewed",
    "insurance_selected",
    "checkout_started",
    "payment_attempted",
    "payment_completed",
]

USER_DIMENSIONS = {
    "source": "utm_source_first",
    "medium": "utm_medium_first",
    "campaign": "utm_campaign_first",
    "language": "language_first",
}

RESERVATION_DIMENSIONS = {
    "source": "utm_source",
    "medium": "utm_medium",
    "campaign": "utm_campaign",
    "language": "language_resolved",
}

MIN_USERS_FOR_TEST = 50
MIN_RESERVATIONS_FOR_PRICE = 30


def clean_segment(value: object) -> str:
    if pd.isna(value):
        return "(unknown)"
    text = str(value).strip()
    return text if text else "(unknown)"


def p_from_z(z: float) -> float:
    if not math.isfinite(z):
        return 1.0
    return math.erfc(abs(z) / math.sqrt(2))


def two_prop_test(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    if n1 <= 0 or n2 <= 0:
        return 0.0, 1.0
    p1 = x1 / n1
    p2 = x2 / n2
    pooled = (x1 + x2) / (n1 + n2)
    se = math.sqrt(max(pooled * (1 - pooled) * (1 / n1 + 1 / n2), 0))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    return z, p_from_z(z)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    p = p_values.fillna(1.0).astype(float).to_numpy()
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted[order[i]] = running
    return pd.Series(np.minimum(adjusted, 1.0), index=p_values.index)


def add_test_fields(rows: list[dict], metric_name: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["q_value"] = bh_adjust(df["p_value"])
    df["significant_fdr_05"] = df["q_value"] < 0.05
    df["direction"] = np.where(df["diff_pp"] > 0, "higher", "lower")
    df["metric"] = metric_name
    return df


def summarize_user_dimension(user: pd.DataFrame, dim_name: str, dim_col: str) -> pd.DataFrame:
    x = user.copy()
    x["segment"] = x[dim_col].map(clean_segment)
    out = x.groupby("segment", dropna=False).agg(
        users=("canonical_user_key", "count"),
        reservation_users=("has_reservation", "sum"),
        paid_users=("has_paid", "sum"),
        insurance_event_users=("has_insurance_selected_event", "sum"),
        insurance_row_users=("has_reservation_insurance_row", "sum"),
        avg_event_count=("event_count", "mean"),
        avg_reservation_count=("reservation_count", "mean"),
    ).reset_index()
    out.insert(0, "dimension", dim_name)
    out["reservation_rate"] = out["reservation_users"] / out["users"]
    out["paid_user_rate"] = out["paid_users"] / out["users"]
    out["paid_after_reservation_rate"] = out["paid_users"] / out["reservation_users"].replace(0, np.nan)
    out["insurance_event_rate_all_users"] = out["insurance_event_users"] / out["users"]
    out["insurance_row_rate_all_users"] = out["insurance_row_users"] / out["users"]
    out["insurance_row_rate_reservation_users"] = out["insurance_row_users"] / out["reservation_users"].replace(0, np.nan)
    return out.sort_values(["dimension", "paid_users", "users"], ascending=[True, False, False])


def funnel_tests(user: pd.DataFrame, dim_name: str, dim_col: str) -> pd.DataFrame:
    total_users = len(user)
    total_by_stage = {
        stage: int((user["funnel_stage_rank"] >= rank).sum())
        for rank, stage in enumerate(STAGE_ORDER, start=1)
    }
    rows: list[dict] = []
    x = user.copy()
    x["segment"] = x[dim_col].map(clean_segment)
    for segment, g in x.groupby("segment", dropna=False):
        n1 = len(g)
        n2 = total_users - n1
        if n1 == 0 or n2 == 0:
            continue
        for rank, stage in enumerate(STAGE_ORDER, start=1):
            reached = int((g["funnel_stage_rank"] >= rank).sum())
            rest_reached = total_by_stage[stage] - reached
            z, p = two_prop_test(reached, n1, rest_reached, n2)
            rate = reached / n1
            rest_rate = rest_reached / n2
            rows.append(
                {
                    "dimension": dim_name,
                    "segment": segment,
                    "stage": stage,
                    "stage_rank": rank,
                    "segment_users": n1,
                    "segment_reached": reached,
                    "segment_rate": rate,
                    "rest_rate": rest_rate,
                    "diff_pp": rate - rest_rate,
                    "z_score": z,
                    "p_value": p,
                    "meets_min_users": n1 >= MIN_USERS_FOR_TEST,
                }
            )
    return add_test_fields(rows, "funnel_stage_reach")


def insurance_tests(user: pd.DataFrame, dim_name: str, dim_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    x = user.copy()
    x["segment"] = x[dim_col].map(clean_segment)
    total_users = len(x)
    total_event = int(x["has_insurance_selected_event"].sum())
    reservers = x[x["has_reservation"]].copy()
    total_reservers = len(reservers)
    total_row = int(reservers["has_reservation_insurance_row"].sum())

    for segment, g in x.groupby("segment", dropna=False):
        n1 = len(g)
        n2 = total_users - n1
        event = int(g["has_insurance_selected_event"].sum())
        rest_event = total_event - event
        z, p = two_prop_test(event, n1, rest_event, n2)
        rows.append(
            {
                "dimension": dim_name,
                "segment": segment,
                "insurance_metric": "event_selected_all_users",
                "denominator": "all_users",
                "segment_n": n1,
                "segment_successes": event,
                "segment_rate": event / n1 if n1 else np.nan,
                "rest_rate": rest_event / n2 if n2 else np.nan,
                "diff_pp": event / n1 - rest_event / n2 if n1 and n2 else np.nan,
                "z_score": z,
                "p_value": p,
                "meets_min_users": n1 >= MIN_USERS_FOR_TEST,
            }
        )

    reservers["segment"] = reservers[dim_col].map(clean_segment)
    for segment, g in reservers.groupby("segment", dropna=False):
        n1 = len(g)
        n2 = total_reservers - n1
        row = int(g["has_reservation_insurance_row"].sum())
        rest_row = total_row - row
        z, p = two_prop_test(row, n1, rest_row, n2)
        rows.append(
            {
                "dimension": dim_name,
                "segment": segment,
                "insurance_metric": "reservation_insurance_row_reservation_users",
                "denominator": "reservation_users",
                "segment_n": n1,
                "segment_successes": row,
                "segment_rate": row / n1 if n1 else np.nan,
                "rest_rate": rest_row / n2 if n2 else np.nan,
                "diff_pp": row / n1 - rest_row / n2 if n1 and n2 else np.nan,
                "z_score": z,
                "p_value": p,
                "meets_min_users": n1 >= MIN_USERS_FOR_TEST,
            }
        )
    return add_test_fields(rows, "insurance_selection")


def insurance_dropoff_tests(user: pd.DataFrame, dim_name: str, dim_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    reservers = user[user["has_reservation"]].copy()
    reservers["segment"] = reservers[dim_col].map(clean_segment)
    total_reservers = len(reservers)

    metrics = [
        (
            "no_insurance_selected_event_among_reservation_users",
            ~reservers["has_insurance_selected_event"],
            "reservation_users",
        ),
        (
            "no_insurance_row_among_reservation_users",
            ~reservers["has_reservation_insurance_row"],
            "reservation_users",
        ),
    ]

    for metric_name, dropoff_mask, denominator in metrics:
        total_dropoffs = int(dropoff_mask.sum())
        for segment, g in reservers.groupby("segment", dropna=False):
            n1 = len(g)
            n2 = total_reservers - n1
            segment_dropoffs = int(dropoff_mask.loc[g.index].sum())
            rest_dropoffs = total_dropoffs - segment_dropoffs
            z, p = two_prop_test(segment_dropoffs, n1, rest_dropoffs, n2)
            segment_rate = segment_dropoffs / n1 if n1 else np.nan
            rest_rate = rest_dropoffs / n2 if n2 else np.nan
            rows.append(
                {
                    "dimension": dim_name,
                    "segment": segment,
                    "dropoff_metric": metric_name,
                    "denominator": denominator,
                    "segment_n": n1,
                    "segment_dropoffs": segment_dropoffs,
                    "segment_dropoff_rate": segment_rate,
                    "rest_dropoff_rate": rest_rate,
                    "diff_pp": segment_rate - rest_rate if n1 and n2 else np.nan,
                    "z_score": z,
                    "p_value": p,
                    "meets_min_users": n1 >= MIN_USERS_FOR_TEST,
                }
            )

    return add_test_fields(rows, "insurance_dropoff")


def price_sensitivity(res: pd.DataFrame, dim_name: str, dim_col: str, q25: float, q75: float) -> pd.DataFrame:
    x = res.copy()
    x["segment"] = x[dim_col].map(clean_segment)
    x["is_paid"] = x["payment_status"].eq("PAID")
    x["price_bucket"] = np.select(
        [x["final_price_base"] <= q25, x["final_price_base"] >= q75],
        ["low_price_q1", "high_price_q4"],
        default="mid_price_q2_q3",
    )
    rows: list[dict] = []
    for segment, g in x.groupby("segment", dropna=False):
        valid_price = g["final_price_base"].notna() & (g["final_price_base"] > 0)
        gv = g[valid_price]
        low = gv[gv["price_bucket"].eq("low_price_q1")]
        high = gv[gv["price_bucket"].eq("high_price_q4")]
        low_paid = int(low["is_paid"].sum())
        high_paid = int(high["is_paid"].sum())
        low_n = len(low)
        high_n = len(high)
        z, p = two_prop_test(high_paid, high_n, low_paid, low_n)
        low_rate = low_paid / low_n if low_n else np.nan
        high_rate = high_paid / high_n if high_n else np.nan
        all_paid = int(g["is_paid"].sum())
        paid_price = g.loc[g["is_paid"] & valid_price, "final_price_base"]
        unpaid_price = g.loc[~g["is_paid"] & valid_price, "final_price_base"]
        rows.append(
            {
                "dimension": dim_name,
                "segment": segment,
                "reservations": len(g),
                "priced_reservations": len(gv),
                "paid_reservations": all_paid,
                "paid_reservation_rate": all_paid / len(g) if len(g) else np.nan,
                "insurance_row_rate_reservations": (g["insurance_row_count"].fillna(0) > 0).mean(),
                "avg_price_base": gv["final_price_base"].mean(),
                "median_price_base": gv["final_price_base"].median(),
                "avg_paid_price_base": paid_price.mean(),
                "avg_unpaid_price_base": unpaid_price.mean(),
                "low_price_reservations": low_n,
                "low_price_paid_rate": low_rate,
                "high_price_reservations": high_n,
                "high_price_paid_rate": high_rate,
                "high_minus_low_paid_rate": high_rate - low_rate if pd.notna(high_rate) and pd.notna(low_rate) else np.nan,
                "price_sensitivity_z": z,
                "price_sensitivity_p_value": p,
                "meets_min_reservations": len(g) >= MIN_RESERVATIONS_FOR_PRICE,
                "meets_min_low_high": low_n >= 10 and high_n >= 10,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["price_sensitivity_q_value"] = bh_adjust(out["price_sensitivity_p_value"])
    out["price_signal"] = np.select(
        [
            out["high_minus_low_paid_rate"].ge(-0.03) & out["meets_min_low_high"],
            out["high_minus_low_paid_rate"].le(-0.08) & out["meets_min_low_high"],
        ],
        ["price_resilient", "price_sensitive"],
        default="inconclusive",
    )
    return out.sort_values(["dimension", "paid_reservations", "reservations"], ascending=[True, False, False])


def build_step_conversion_table(funnel: pd.DataFrame) -> pd.DataFrame:
    target_segments = [
        ("campaign", "23754383997"),
        ("campaign", "23735454208"),
        ("campaign", "23205597293"),
        ("campaign", "23688781100"),
        ("medium", "demandgen"),
        ("channel", "fb / paid"),
        ("campaign", "seoul_en_2026q2"),
        ("campaign", "seoul_zh_2026q2"),
    ]
    stage_labels = {
        "identified_visit": "식별 방문",
        "search_started": "검색 시작",
        "car_clicked": "차량 클릭",
        "booking_started": "예약/가격확인",
        "payment_attempted": "결제 시도",
        "payment_completed": "결제 완료",
    }
    rows = []
    for dimension, segment in target_segments:
        x = funnel[
            funnel["dimension"].eq(dimension)
            & funnel["segment"].eq(segment)
            & funnel["stage"].isin(stage_labels)
        ].sort_values("stage_rank")
        if x.empty:
            continue
        counts = dict(zip(x["stage"], x["segment_reached"]))
        users = int(x["segment_users"].iloc[0])
        previous = users
        row = {"구분": dimension, "세그먼트": segment, "유저 수": users}
        for stage, label in stage_labels.items():
            reached = int(counts.get(stage, 0))
            row[f"{label} 수"] = reached
            row[f"{label} 전환율"] = reached / previous if previous else np.nan
            previous = reached
        row["예약/가격확인 → 결제완료"] = (
            row["결제 완료 수"] / row["예약/가격확인 수"]
            if row["예약/가격확인 수"]
            else np.nan
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    return out[
        [
            "구분",
            "세그먼트",
            "유저 수",
            "식별 방문 전환율",
            "검색 시작 전환율",
            "차량 클릭 전환율",
            "예약/가격확인 전환율",
            "결제 시도 전환율",
            "결제 완료 전환율",
            "예약/가격확인 → 결제완료",
        ]
    ].rename(
        columns={
            "식별 방문 전환율": "전체 → 식별",
            "검색 시작 전환율": "식별 → 검색",
            "차량 클릭 전환율": "검색 → 차량클릭",
            "예약/가격확인 전환율": "차량클릭 → 예약/가격확인",
            "결제 시도 전환율": "예약/가격확인 → 결제시도",
            "결제 완료 전환율": "결제시도 → 결제완료",
        }
    )


def write_html_report(
    summary: pd.DataFrame,
    funnel: pd.DataFrame,
    insurance: pd.DataFrame,
    insurance_dropoff: pd.DataFrame,
    price: pd.DataFrame,
    path: Path,
) -> None:
    top_funnel = funnel[
        funnel["meets_min_users"] & funnel["significant_fdr_05"] & funnel["stage"].isin(
            ["identified_visit", "search_started", "car_clicked", "booking_started", "payment_attempted", "payment_completed"]
        )
    ].copy()
    top_funnel["abs_diff_pp"] = top_funnel["diff_pp"].abs()
    top_funnel = top_funnel.sort_values("abs_diff_pp", ascending=False).head(60)
    step_conversion = build_step_conversion_table(funnel)

    top_ins = insurance[
        insurance["meets_min_users"] & insurance["significant_fdr_05"]
    ].copy()
    top_ins["abs_diff_pp"] = top_ins["diff_pp"].abs()
    top_ins = top_ins.sort_values("abs_diff_pp", ascending=False).head(60)

    top_ins_dropoff = insurance_dropoff[
        insurance_dropoff["meets_min_users"] & insurance_dropoff["significant_fdr_05"]
    ].copy()
    top_ins_dropoff["abs_diff_pp"] = top_ins_dropoff["diff_pp"].abs()
    top_ins_dropoff = top_ins_dropoff.sort_values("abs_diff_pp", ascending=False).head(80)

    top_price = price[
        price["meets_min_reservations"] & price["meets_min_low_high"]
    ].copy()
    top_price["abs_price_gap"] = top_price["high_minus_low_paid_rate"].abs()
    top_price = top_price.sort_values(["price_signal", "abs_price_gap"], ascending=[True, False]).head(80)

    css = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 32px; color: #202124; }
    h1, h2, h3 { margin-bottom: 8px; }
    p, li { max-width: 1080px; line-height: 1.5; }
    .note { background: #f8fafc; border-left: 4px solid #64748b; padding: 12px 14px; margin: 12px 0 20px; }
    .warn { background: #fff7ed; border-left: 4px solid #f97316; padding: 12px 14px; margin: 12px 0 20px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; margin: 12px 0 24px; }
    .box { border: 1px solid #ddd; padding: 12px; border-radius: 6px; background: #fff; }
    .box h3 { margin-top: 0; font-size: 15px; }
    table { border-collapse: collapse; font-size: 12px; margin: 12px 0 28px; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { text-align: left; }
    th { background: #f3f4f6; position: sticky; top: 0; }
    code { background: #f3f4f6; padding: 1px 4px; border-radius: 4px; }
    </style>
    """
    sections = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        css,
        "</head><body>",
        "<h1>Segment Funnel, Insurance, Price Sensitivity Diagnostics</h1>",
        "<p>분석 대상은 <code>identity_level != reservation_id</code>인 canonical user와 해당 예약입니다. "
        "비율 차이는 세그먼트 vs 나머지 전체의 two-proportion z-test로 계산하고 FDR 5% 기준을 표시했습니다. "
        "가격민감도는 전역 가격 하위 25% 예약과 상위 25% 예약의 결제 완료율 차이로 봤습니다.</p>",
        "<div class='warn'><strong>중요한 해석 주의사항.</strong> "
        "<code>language=(unknown)</code>은 성과가 좋은 언어가 아니라, 이벤트가 붙지 않은 예약 기반 customer 유저가 몰린 "
        "attribution 실패 그룹입니다. 또한 예약 row의 UTM은 대부분 <code>(unknown)</code>이라 source/medium/campaign 기준 "
        "가격민감도는 현재 데이터만으로 강하게 해석하지 않는 것이 안전합니다.</div>",
        "<h2>Analysis Steps</h2>",
        "<div class='grid'>",
        "<div class='box'><h3>1. 데이터 정리</h3>"
        "<p><code>user_funnel_state.csv</code>와 <code>reservation_funnel_state.csv</code>를 읽고, "
        "<code>identity_level == reservation_id</code>인 단위는 유저 분석에서 제외했습니다. 이 단위는 customer_id 없이 "
        "reservation_id만 있는 미식별 예약 단위라 유저 전환율을 과장할 수 있습니다.</p></div>",
        "<div class='box'><h3>2. 세그먼트 정의</h3>"
        "<p>유저 기준은 <code>source</code>, <code>medium</code>, <code>campaign</code>, <code>language</code>, "
        "<code>channel = source / medium</code>입니다. 결측이나 빈 문자열은 <code>(unknown)</code>으로 통일했습니다.</p></div>",
        "<div class='box'><h3>3. 유저 퍼널 계산</h3>"
        "<p>각 canonical user의 <code>funnel_stage_rank</code>가 특정 단계 rank 이상이면 그 단계에 도달한 것으로 봤습니다. "
        "즉 단계별 수치는 순수 이벤트 카운트가 아니라 누적 도달 유저 수입니다.</p></div>",
        "<div class='box'><h3>4. 유의성 검정</h3>"
        "<p>각 세그먼트의 단계 도달률을 나머지 전체와 비교했습니다. 예를 들어 app 결제완료율은 app 유저와 non-app 유저의 "
        "결제완료율 차이를 two-proportion z-test로 계산하고, 여러 비교 문제는 Benjamini-Hochberg FDR로 보정했습니다.</p></div>",
        "<div class='box'><h3>5. 보험 선택 분석</h3>"
        "<p>보험은 두 가지로 나눠 봤습니다. <code>has_insurance_selected_event</code>는 화면/이벤트상 선택 행동이고, "
        "<code>has_reservation_insurance_row</code>는 실제 예약 DB에 보험 row가 붙었는지입니다. 세그먼트 차이는 이벤트 기준이 더 잘 드러납니다.</p></div>",
        "<div class='box'><h3>6. 가격민감도 분석</h3>"
        "<p>예약의 <code>final_price_base</code> 기준으로 전체 하위 25%를 저가, 상위 25%를 고가로 정의했습니다. "
        "고가 결제율 - 저가 결제율이 크게 음수면 가격민감도가 높은 것으로 봤습니다.</p></div>",
        "</div>",
        "<h2>Column Guide</h2>",
        "<ul>",
        "<li><code>segment_rate</code>: 해당 세그먼트 안에서의 도달률/선택률/결제율입니다.</li>",
        "<li><code>rest_rate</code>: 해당 세그먼트를 제외한 나머지 전체의 같은 비율입니다.</li>",
        "<li><code>diff_pp</code>: <code>segment_rate - rest_rate</code>입니다. 양수면 세그먼트가 더 높고, 음수면 더 낮습니다.</li>",
        "<li><code>q_value</code>: 여러 세그먼트와 단계를 동시에 비교한 뒤 FDR 보정한 p-value입니다. 0.05 미만이면 유의한 차이로 표시했습니다.</li>",
        "<li><code>meets_min_users</code>, <code>meets_min_reservations</code>: 작은 표본의 과해석을 막기 위한 최소 표본 기준입니다.</li>",
        "</ul>",
        "<h2>Notion Copy Table: Step Conversion Rates</h2>",
        "<p>아래 표는 노션에 복사하기 쉽도록 주요 캠페인/채널의 구간 전환율만 정리한 표입니다. "
        "브라우저에서 표 전체를 드래그해서 복사하면 노션에 표 형태로 붙여넣을 수 있습니다. "
        "<code>-</code>로 보이는 빈 값은 이전 단계 도달자가 0명이라 계산할 수 없는 구간입니다.</p>",
        step_conversion.to_html(index=False, escape=True, na_rep="-", float_format=lambda x: f"{x:.1%}" if abs(x) <= 1 else f"{x:,.0f}"),
        "<h2>Segment Summary</h2>",
        "<p>이 표는 각 세그먼트의 기본 성과판입니다. 먼저 유저 수가 충분한지 확인한 뒤, "
        "<code>reservation_rate</code>, <code>paid_user_rate</code>, <code>paid_after_reservation_rate</code>, "
        "<code>insurance_event_rate_all_users</code>를 함께 봅니다. 방문 규모가 큰 세그먼트와 실제 결제 효율이 좋은 세그먼트는 다를 수 있습니다.</p>",
        summary.head(80).to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "<h2>Top Significant Funnel Differences</h2>",
        "<p>이 표는 퍼널 단계별로 세그먼트와 나머지 전체의 도달률 차이가 큰 항목입니다. "
        "<code>booking_started</code>는 예약 생성까지의 힘, <code>payment_attempted</code>는 결제 시도까지의 힘, "
        "<code>payment_completed</code>는 최종 전환 품질을 보여줍니다. <code>direction=higher</code>는 해당 세그먼트가 나머지보다 "
        "높다는 뜻이고, <code>lower</code>는 낮다는 뜻입니다.</p>",
        "<div class='note'><strong>읽는 법.</strong> 예를 들어 <code>source=app</code>의 <code>payment_completed</code> "
        "diff_pp가 +0.148이면 app 유저의 결제완료율이 non-app 유저보다 약 14.8%p 높다는 뜻입니다. "
        "반대로 <code>google / demandgen</code>이 음수로 크면 해당 유입은 같은 단계에서 나머지보다 약하다는 뜻입니다.</div>",
        top_funnel.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "<h2>Top Significant Insurance Differences</h2>",
        "<p>이 표는 보험 선택 관련 세그먼트 차이입니다. <code>event_selected_all_users</code>는 모든 유저 대비 보험 선택 이벤트가 있었는지, "
        "<code>reservation_insurance_row_reservation_users</code>는 예약 유저 중 실제 보험 row가 있는지를 봅니다. "
        "현재 데이터에서는 예약 row 기준 보험 부착률이 대부분 높아 차이가 작고, 이벤트 기준이 세그먼트별 관심도 차이를 더 잘 보여줍니다.</p>",
        "<div class='note'><strong>활용 포인트.</strong> 보험 이벤트율이 높은 세그먼트는 보험 메시지, 보장 내용, 결제 전 안심 요소에 반응할 가능성이 높습니다. "
        "보험 이벤트율은 낮지만 결제율도 낮은 세그먼트는 보험 설명 노출 자체가 부족했거나, 예약 플로우 깊이까지 도달하지 못했을 수 있습니다.</div>",
        top_ins.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "<h2>Top Significant Insurance Dropoff Differences</h2>",
        "<p>이 표는 예약 단계에 도달한 유저 중 보험 선택이 없었던 비율을 비교합니다. "
        "<code>no_insurance_selected_event_among_reservation_users</code>는 예약 유저 중 보험 선택 이벤트가 없었던 비율이고, "
        "<code>no_insurance_row_among_reservation_users</code>는 예약 유저 중 실제 보험 row가 없었던 비율입니다. "
        "<code>diff_pp</code>가 양수이면 해당 세그먼트의 보험 이탈률이 나머지보다 높고, 음수이면 낮습니다.</p>",
        "<div class='warn'><strong>주의.</strong> 보험 선택 이벤트가 누락될 수 있으므로 이벤트 기준 이탈률은 tracking 품질의 영향을 받습니다. "
        "실제 보험 row 기준은 더 보수적이지만 대부분 row가 붙어 있어 세그먼트 차이가 작게 나옵니다.</div>",
        top_ins_dropoff.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "<h2>Price Sensitivity Signals</h2>",
        "<p>이 표는 예약 가격대별 결제율 차이를 봅니다. <code>low_price_paid_rate</code>는 전체 가격 하위 25% 예약의 결제완료율이고, "
        "<code>high_price_paid_rate</code>는 전체 가격 상위 25% 예약의 결제완료율입니다. "
        "<code>high_minus_low_paid_rate</code>가 -0.08 이하이면 가격민감 신호, -0.03 이상이면 가격 방어 신호로 분류했습니다.</p>",
        "<div class='warn'><strong>가격 분석 한계.</strong> 예약 row의 source/medium/campaign 값이 대부분 <code>(unknown)</code>입니다. "
        "따라서 가격민감도는 현재 산출물에서는 language 중심으로 해석하고, 캠페인별 가격민감도는 예약 UTM attribution 보강 후 다시 보는 것이 좋습니다.</div>",
        top_price.to_html(index=False, escape=True, float_format=lambda x: f"{x:.4f}"),
        "<h2>Recommended Next Checks</h2>",
        "<ul>",
        "<li><strong>language unknown 보정:</strong> customer_id와 Mixpanel distinct_id/session_id를 다시 bridge해서 언어 attribution을 보강합니다.</li>",
        "<li><strong>예약 UTM 보강:</strong> reservation row에 user-level first-touch 또는 last-touch UTM을 붙여 campaign별 가격민감도를 재계산합니다.</li>",
        "<li><strong>app 벤치마크:</strong> app 세그먼트의 결제완료율과 보험 이벤트율이 높은 이유를 플로우, 재방문, 로그인 상태, 결제수단 관점에서 분해합니다.</li>",
        "<li><strong>zh 가격 테스트:</strong> zh 세그먼트는 고가 구간 결제율 저하가 뚜렷하므로 할인/보장/환불/보험 메시지를 별도로 테스트합니다.</li>",
        "</ul>",
        "</body></html>",
    ]
    path.write_text("\n".join(sections), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    user = pd.read_csv(PROCESSED / "user_funnel_state.csv", low_memory=False)
    res = pd.read_csv(PROCESSED / "reservation_funnel_state.csv", low_memory=False)

    user = user[user["identity_level"].ne("reservation_id")].copy()
    res = res[res["identity_level"].ne("reservation_id")].copy()

    numeric_user_cols = [
        "funnel_stage_rank",
        "event_count",
        "reservation_count",
        "paid_reservation_count",
        "insurance_row_count",
    ]
    for col in numeric_user_cols:
        user[col] = pd.to_numeric(user[col], errors="coerce").fillna(0)
    user["has_reservation"] = user["reservation_count"] > 0
    user["has_paid"] = user["paid_reservation_count"] > 0
    user["has_insurance_selected_event"] = user["has_insurance_selected_event"].fillna(False).astype(bool)
    user["has_reservation_insurance_row"] = user["has_reservation_insurance_row"].fillna(False).astype(bool)
    user["channel_medium_first"] = user["utm_source_first"].map(clean_segment) + " / " + user["utm_medium_first"].map(clean_segment)
    USER_DIMENSIONS_WITH_CHANNEL = dict(USER_DIMENSIONS)
    USER_DIMENSIONS_WITH_CHANNEL["channel"] = "channel_medium_first"

    language_map = user.set_index("canonical_user_key")["language_first"].map(clean_segment).to_dict()
    res["language_resolved"] = res["canonical_user_key"].map(language_map).fillna("(unknown)")
    res["final_price_base"] = pd.to_numeric(res["final_price_base"], errors="coerce")
    res["insurance_row_count"] = pd.to_numeric(res["insurance_row_count"], errors="coerce").fillna(0)
    res["channel"] = res["utm_source"].map(clean_segment) + " / " + res["utm_medium"].map(clean_segment)
    RESERVATION_DIMENSIONS_WITH_CHANNEL = dict(RESERVATION_DIMENSIONS)
    RESERVATION_DIMENSIONS_WITH_CHANNEL["channel"] = "channel"

    user_summary = pd.concat(
        [
            summarize_user_dimension(user, dim_name, dim_col)
            for dim_name, dim_col in USER_DIMENSIONS_WITH_CHANNEL.items()
        ],
        ignore_index=True,
    )
    funnel = pd.concat(
        [
            funnel_tests(user, dim_name, dim_col)
            for dim_name, dim_col in USER_DIMENSIONS_WITH_CHANNEL.items()
        ],
        ignore_index=True,
    )
    insurance = pd.concat(
        [
            insurance_tests(user, dim_name, dim_col)
            for dim_name, dim_col in USER_DIMENSIONS_WITH_CHANNEL.items()
        ],
        ignore_index=True,
    )
    insurance_dropoff = pd.concat(
        [
            insurance_dropoff_tests(user, dim_name, dim_col)
            for dim_name, dim_col in USER_DIMENSIONS_WITH_CHANNEL.items()
        ],
        ignore_index=True,
    )

    valid_price = res["final_price_base"].notna() & (res["final_price_base"] > 0)
    q25 = float(res.loc[valid_price, "final_price_base"].quantile(0.25))
    q75 = float(res.loc[valid_price, "final_price_base"].quantile(0.75))
    price = pd.concat(
        [
            price_sensitivity(res, dim_name, dim_col, q25, q75)
            for dim_name, dim_col in RESERVATION_DIMENSIONS_WITH_CHANNEL.items()
        ],
        ignore_index=True,
    )

    user_summary.to_csv(OUT / "segment_user_summary.csv", index=False, encoding="utf-8-sig")
    funnel.to_csv(OUT / "segment_funnel_significance.csv", index=False, encoding="utf-8-sig")
    insurance.to_csv(OUT / "segment_insurance_significance.csv", index=False, encoding="utf-8-sig")
    insurance_dropoff.to_csv(OUT / "segment_insurance_dropoff_significance.csv", index=False, encoding="utf-8-sig")
    price.to_csv(OUT / "segment_price_sensitivity.csv", index=False, encoding="utf-8-sig")
    build_step_conversion_table(funnel).to_csv(OUT / "notion_step_conversion_table.csv", index=False, encoding="utf-8-sig")

    diagnostics = {
        "analysis_users": int(len(user)),
        "analysis_reservations": int(len(res)),
        "price_q25_final_price_base": q25,
        "price_q75_final_price_base": q75,
        "min_users_for_test": MIN_USERS_FOR_TEST,
        "min_reservations_for_price": MIN_RESERVATIONS_FOR_PRICE,
        "dimensions": list(USER_DIMENSIONS_WITH_CHANNEL.keys()),
        "notes": [
            "language=(unknown) is structurally biased because many customer_id-only reservation users have event_count=0.",
            "price sensitivity compares paid rate in global low-price Q1 reservations vs high-price Q4 reservations.",
        ],
    }
    (OUT / "segment_diagnostics_summary.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_html_report(
        user_summary,
        funnel,
        insurance,
        insurance_dropoff,
        price,
        OUT / "segment_diagnostics_report.html",
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
