from __future__ import annotations

import html as html_lib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "analysis4"
sys.path.insert(0, str(OUT_DIR))

from holding_period_analysis import clean_reservations, load_data  # noqa: E402


RECENT_MONTHS = 3
EARLY_MONTHS = 3
DEFAULT_PURCHASE_PRICE_BY_CLASS = {
    "URBAN": 20_000_000,
    "COMPACT": 25_000_000,
    "MIDSIZE": 35_000_000,
    "SUV": 40_000_000,
    "RV": 45_000_000,
    "EV": 45_000_000,
    "IMPORTED": 75_000_000,
    "SUPERCAR": 150_000_000,
}


def fmt_krw(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}원"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def fmt_num(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.0f}"


def fmt_score(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.0f}"


def month_age_years(month: pd.Series, model_year: pd.Series) -> pd.Series:
    month_start = pd.to_datetime(month.astype(str) + "-01", errors="coerce")
    model_start = pd.to_datetime(model_year.astype("Int64").astype(str) + "-01-01", errors="coerce")
    return ((month_start - model_start).dt.days / 365.25).clip(lower=0)


def bar_chart(df: pd.DataFrame, label_col: str, value_col: str, formatter, color: str) -> str:
    if df.empty:
        return '<div class="note">표시할 데이터가 없습니다.</div>'
    values = pd.to_numeric(df[value_col], errors="coerce").fillna(0)
    max_value = values.max()
    max_value = max_value if max_value > 0 else 1
    rows = []
    for _, row in df.iterrows():
        value = float(row[value_col] or 0)
        width = max(1, min(100, value / max_value * 100))
        label = html_lib.escape(str(row[label_col]))
        rows.append(
            f'<div class="bar-row"><div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%; background:{color};"></div></div>'
            f'<div class="bar-value">{formatter(value)}</div></div>'
        )
    return "\n".join(rows)


def build_depreciation_payload(monthly: pd.DataFrame) -> tuple[str, int]:
    chart = monthly.copy()
    chart["age_years"] = month_age_years(chart["month"], chart["model_year_latest"])
    chart["age_bucket_quarter"] = (chart["age_years"] * 4).round() / 4
    chart = chart[chart["fleet_vehicle_days"] >= 20].copy()
    grouped = (
        chart.groupby(["model_family", "vehicle_class", "age_bucket_quarter"], as_index=False)
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
            reservation_count=("reservation_count", "sum"),
        )
    )
    grouped["revenue_per_fleet_day"] = np.where(
        grouped["fleet_vehicle_days"] > 0,
        grouped["revenue_krw"] / grouped["fleet_vehicle_days"],
        np.nan,
    )
    grouped["utilization_rate"] = np.where(
        grouped["fleet_vehicle_days"] > 0,
        grouped["booked_vehicle_days"] / grouped["fleet_vehicle_days"],
        np.nan,
    )
    family_totals = (
        grouped.groupby(["model_family", "vehicle_class"], as_index=False)
        .agg(total_revenue_krw=("revenue_krw", "sum"))
        .sort_values("total_revenue_krw", ascending=False)
    )
    order = {
        (row.model_family, row.vehicle_class): idx
        for idx, row in enumerate(family_totals.itertuples(index=False))
    }
    payload = []
    for keys, frame in grouped.groupby(["model_family", "vehicle_class"]):
        frame = frame.sort_values("age_bucket_quarter")
        if frame["age_bucket_quarter"].nunique() < 2:
            continue
        vehicle_class = str(keys[1])
        default_price = DEFAULT_PURCHASE_PRICE_BY_CLASS.get(vehicle_class, 35_000_000)
        payload.append(
            {
                "modelFamily": str(keys[0]),
                "vehicleClass": vehicle_class,
                "sortOrder": order.get(keys, 9999),
                "defaultPurchasePriceKrw": default_price,
                "points": [
                    {
                        "age": round(float(row.age_bucket_quarter), 2),
                        "revenuePerFleetDay": round(float(row.revenue_per_fleet_day), 1),
                        "fleetVehicleDays": round(float(row.fleet_vehicle_days), 1),
                        "reservationCount": round(float(row.reservation_count), 1),
                        "utilizationRate": round(float(row.utilization_rate), 4),
                    }
                    for row in frame.itertuples(index=False)
                    if pd.notna(row.revenue_per_fleet_day)
                ],
            }
        )
    payload.sort(key=lambda item: item["sortOrder"])
    return json.dumps(payload, ensure_ascii=False), len(payload)


def depreciation_chart_section(monthly: pd.DataFrame) -> str:
    payload_json, family_count = build_depreciation_payload(monthly)
    return f"""
  <h2>차종별 매출·감가 시뮬레이션</h2>
  <p class="note">감가상각 원장은 아직 없으므로 아래 그래프의 감가선은 클래스별 기본 차량가와 연차별 감가율을 적용한 추정치입니다.
  매출선은 실제 월별 보유일당 매출을 차령 0.25년 단위로 묶은 값이며, 감가 가정은 입력값을 바꿔 즉시 재계산할 수 있습니다.</p>
  <div class="depreciation-tool">
    <div class="control-row">
      <label>차종 <select id="depreciationModelSelect"></select></label>
      <label>차량가 <input id="purchasePriceInput" type="number" min="1000000" step="1000000"></label>
      <label>1년차 <input id="rateYear1Input" type="number" min="0" max="80" step="1" value="18">%</label>
      <label>2년차 <input id="rateYear2Input" type="number" min="0" max="80" step="1" value="14">%</label>
      <label>3년차 <input id="rateYear3Input" type="number" min="0" max="80" step="1" value="11">%</label>
      <label>이후 <input id="rateLaterInput" type="number" min="0" max="80" step="1" value="8">%</label>
      <label>잔존하한 <input id="floorPctInput" type="number" min="0" max="90" step="5" value="25">%</label>
    </div>
    <div class="chart-wrap">
      <svg id="depreciationChart" viewBox="0 0 980 420" role="img" aria-label="차종별 매출 감가 비교 그래프"></svg>
    </div>
    <div class="legend">
      <span><i style="background:#2563eb"></i> 보유일당 매출</span>
      <span><i style="background:#dc2626"></i> 추정 감가/일</span>
      <span><i style="background:#166534"></i> 매출-감가</span>
    </div>
    <div id="depreciationSummary" class="chart-summary"></div>
  </div>
  <script>
    const depreciationData = {payload_json};
    const krw = new Intl.NumberFormat("ko-KR");
    const select = document.getElementById("depreciationModelSelect");
    const priceInput = document.getElementById("purchasePriceInput");
    const inputs = [
      priceInput,
      document.getElementById("rateYear1Input"),
      document.getElementById("rateYear2Input"),
      document.getElementById("rateYear3Input"),
      document.getElementById("rateLaterInput"),
      document.getElementById("floorPctInput"),
    ];
    const svg = document.getElementById("depreciationChart");
    const summary = document.getElementById("depreciationSummary");

    function populateOptions() {{
      depreciationData
        .slice()
        .sort((a, b) => a.sortOrder - b.sortOrder)
        .forEach((item, index) => {{
          const option = document.createElement("option");
          option.value = index;
          option.textContent = `${{item.modelFamily}} (${{item.vehicleClass}})`;
          select.appendChild(option);
        }});
    }}

    function valueAtAge(price, age, rates, floorPct) {{
      let remainingPct = 1;
      let cursor = 0;
      const segments = [
        [1, rates.year1],
        [2, rates.year2],
        [3, rates.year3],
      ];
      for (const [end, rate] of segments) {{
        if (age > cursor) {{
          remainingPct -= Math.min(age, end) - cursor > 0 ? (Math.min(age, end) - cursor) * rate : 0;
          cursor = end;
        }}
      }}
      if (age > 3) remainingPct -= (age - 3) * rates.later;
      return Math.max(price * remainingPct, price * floorPct);
    }}

    function depreciationPerDay(price, age, rates, floorPct) {{
      const nextAge = age + 1 / 12;
      const monthLoss = valueAtAge(price, age, rates, floorPct) - valueAtAge(price, nextAge, rates, floorPct);
      return Math.max(0, monthLoss / (365.25 / 12));
    }}

    function polyline(points, xScale, yScale, key) {{
      return points.map((point) => `${{xScale(point.age).toFixed(1)}},${{yScale(point[key]).toFixed(1)}}`).join(" ");
    }}

    function drawChart() {{
      const item = depreciationData[Number(select.value || 0)];
      if (!item) return;
      if (document.activeElement !== priceInput || !priceInput.value) {{
        priceInput.value = item.defaultPurchasePriceKrw;
      }}
      const price = Number(priceInput.value || item.defaultPurchasePriceKrw);
      const rates = {{
        year1: Number(document.getElementById("rateYear1Input").value || 0) / 100,
        year2: Number(document.getElementById("rateYear2Input").value || 0) / 100,
        year3: Number(document.getElementById("rateYear3Input").value || 0) / 100,
        later: Number(document.getElementById("rateLaterInput").value || 0) / 100,
      }};
      const floorPct = Number(document.getElementById("floorPctInput").value || 0) / 100;
      const points = item.points
        .map((point) => {{
          const depreciation = depreciationPerDay(price, point.age, rates, floorPct);
          return {{
            ...point,
            depreciationPerDay: depreciation,
            netPerDay: point.revenuePerFleetDay - depreciation,
          }};
        }})
        .sort((a, b) => a.age - b.age);
      const margin = {{ left: 70, right: 34, top: 28, bottom: 54 }};
      const width = 980;
      const height = 420;
      const chartWidth = width - margin.left - margin.right;
      const chartHeight = height - margin.top - margin.bottom;
      const maxAge = Math.max(1, ...points.map((point) => point.age));
      const maxValue = Math.max(1000, ...points.flatMap((point) => [
        point.revenuePerFleetDay,
        point.depreciationPerDay,
        Math.max(0, point.netPerDay),
      ]));
      const xScale = (age) => margin.left + (age / maxAge) * chartWidth;
      const yScale = (value) => margin.top + chartHeight - (value / maxValue) * chartHeight;
      const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => Math.round(maxValue * ratio / 1000) * 1000);
      const ageTicks = Array.from(new Set([0, Math.ceil(maxAge / 4), Math.ceil(maxAge / 2), Math.ceil(maxAge * 3 / 4), Math.ceil(maxAge)]));
      const sellPoint = points.find((point) => point.age >= 1 && point.revenuePerFleetDay <= point.depreciationPerDay);
      const worstPoint = points.reduce((best, point) => !best || point.netPerDay < best.netPerDay ? point : best, null);
      const marker = sellPoint || worstPoint;

      let markup = `<rect x="0" y="0" width="${{width}}" height="${{height}}" fill="#fff"/>`;
      ticks.forEach((tick) => {{
        const y = yScale(tick);
        markup += `<line x1="${{margin.left}}" y1="${{y}}" x2="${{width - margin.right}}" y2="${{y}}" stroke="#edf2f7"/>`;
        markup += `<text x="${{margin.left - 10}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#52606d">${{krw.format(tick)}}원</text>`;
      }});
      ageTicks.forEach((tick) => {{
        const x = xScale(tick);
        markup += `<line x1="${{x}}" y1="${{margin.top}}" x2="${{x}}" y2="${{height - margin.bottom}}" stroke="#f5f7fa"/>`;
        markup += `<text x="${{x}}" y="${{height - 20}}" text-anchor="middle" font-size="11" fill="#52606d">${{tick}}년</text>`;
      }});
      markup += `<line x1="${{margin.left}}" y1="${{height - margin.bottom}}" x2="${{width - margin.right}}" y2="${{height - margin.bottom}}" stroke="#9fb3c8"/>`;
      markup += `<line x1="${{margin.left}}" y1="${{margin.top}}" x2="${{margin.left}}" y2="${{height - margin.bottom}}" stroke="#9fb3c8"/>`;
      markup += `<polyline points="${{polyline(points, xScale, yScale, "revenuePerFleetDay")}}" fill="none" stroke="#2563eb" stroke-width="3"/>`;
      markup += `<polyline points="${{polyline(points, xScale, yScale, "depreciationPerDay")}}" fill="none" stroke="#dc2626" stroke-width="3"/>`;
      markup += `<polyline points="${{polyline(points, xScale, yScale, "netPerDay")}}" fill="none" stroke="#166534" stroke-width="3" stroke-dasharray="6 5"/>`;
      points.forEach((point) => {{
        markup += `<circle cx="${{xScale(point.age)}}" cy="${{yScale(point.revenuePerFleetDay)}}" r="3.5" fill="#2563eb"><title>${{point.age}}년: 매출 ${{krw.format(Math.round(point.revenuePerFleetDay))}}원/일, 감가 ${{krw.format(Math.round(point.depreciationPerDay))}}원/일</title></circle>`;
      }});
      if (marker) {{
        const x = xScale(marker.age);
        markup += `<line x1="${{x}}" y1="${{margin.top}}" x2="${{x}}" y2="${{height - margin.bottom}}" stroke="#111827" stroke-width="1.5" stroke-dasharray="4 4"/>`;
        markup += `<text x="${{Math.min(x + 8, width - 230)}}" y="${{margin.top + 18}}" font-size="12" fill="#111827">매각 검토: ${{marker.age.toFixed(1)}}년</text>`;
      }}
      svg.innerHTML = markup;

      const latest = points[points.length - 1];
      const recommendation = sellPoint
        ? `${{sellPoint.age.toFixed(1)}}년 부근부터 보유일당 매출이 추정 감가/일 이하로 내려갑니다.`
        : `관측된 차령 구간에서는 보유일당 매출이 추정 감가/일보다 높습니다.`;
      summary.innerHTML = `
        <strong>${{item.modelFamily}}</strong> ${{
          recommendation
        }} 최근 관측점 기준 매출 ${{krw.format(Math.round(latest.revenuePerFleetDay))}}원/일,
        감가 ${{krw.format(Math.round(latest.depreciationPerDay))}}원/일,
        순기여 ${{krw.format(Math.round(latest.netPerDay))}}원/일입니다.
      `;
    }}

    populateOptions();
    select.value = "0";
    priceInput.value = depreciationData[0]?.defaultPurchasePriceKrw || 35000000;
    select.addEventListener("change", () => {{
      priceInput.value = depreciationData[Number(select.value)]?.defaultPurchasePriceKrw || 35000000;
      drawChart();
    }});
    inputs.forEach((input) => input.addEventListener("input", drawChart));
    drawChart();
  </script>
  <p class="note">표시 가능 차종: {family_count:,}개. 매각 검토 지점은 실제 매각가가 아니라 입력한 감가 가정으로 계산한 기준선입니다.</p>
"""


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = pd.read_csv(OUT_DIR / "daily_age_revenue.csv", parse_dates=["date"])
    reservations_raw, _, _ = load_data()
    reservations = clean_reservations(reservations_raw)
    return daily, reservations


def read_monthly_input() -> pd.DataFrame:
    try:
        daily, reservations = read_inputs()
    except FileNotFoundError:
        return pd.read_csv(OUT_DIR / "extra_monthly_family_year.csv")
    return build_monthly_family_year(daily, reservations)


def build_monthly_family_year(daily: pd.DataFrame, reservations: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["month"] = daily["date"].dt.to_period("M").astype(str)
    monthly = (
        daily.groupby(["month", "model_family", "vehicle_class", "model_year_latest"], as_index=False)
        .agg(
            fleet_vehicle_days=("fleet_vehicle_days", "sum"),
            booked_vehicle_days=("booked_vehicle_days", "sum"),
            revenue_krw=("revenue_krw", "sum"),
        )
    )
    res = (
        reservations.groupby(
            ["pickup_month", "model_family", "vehicle_class", "model_year_latest"],
            as_index=False,
        )
        .agg(
            reservation_count=("reservation_key", "nunique"),
            unique_joined_users=("user_key", "nunique"),
            avg_reservation_price_krw=("final_price_krw", "mean"),
        )
        .rename(columns={"pickup_month": "month"})
    )
    monthly = monthly.merge(
        res,
        on=["month", "model_family", "vehicle_class", "model_year_latest"],
        how="left",
    )
    monthly[["reservation_count", "unique_joined_users"]] = monthly[
        ["reservation_count", "unique_joined_users"]
    ].fillna(0)
    monthly["utilization_rate"] = np.where(
        monthly["fleet_vehicle_days"] > 0,
        monthly["booked_vehicle_days"] / monthly["fleet_vehicle_days"],
        np.nan,
    )
    monthly["revenue_per_fleet_day"] = np.where(
        monthly["fleet_vehicle_days"] > 0,
        monthly["revenue_krw"] / monthly["fleet_vehicle_days"],
        np.nan,
    )
    family_total = (
        monthly.groupby(["month", "model_family"], as_index=False)
        .agg(family_month_revenue_krw=("revenue_krw", "sum"))
    )
    monthly = monthly.merge(family_total, on=["month", "model_family"], how="left")
    monthly["family_revenue_share"] = np.where(
        monthly["family_month_revenue_krw"] > 0,
        monthly["revenue_krw"] / monthly["family_month_revenue_krw"],
        np.nan,
    )
    return monthly


def window_stats(group: pd.DataFrame) -> dict[str, float]:
    active = group[group["fleet_vehicle_days"] > 0].sort_values("month")
    if active.empty:
        return {}
    early = active.head(EARLY_MONTHS)
    recent = active.tail(RECENT_MONTHS)
    return {
        "first_month": active["month"].min(),
        "last_month": active["month"].max(),
        "active_months": active["month"].nunique(),
        "early_revenue_krw": early["revenue_krw"].sum(),
        "recent_revenue_krw": recent["revenue_krw"].sum(),
        "early_reservations": early["reservation_count"].sum(),
        "recent_reservations": recent["reservation_count"].sum(),
        "early_revenue_per_fleet_day": early["revenue_krw"].sum()
        / early["fleet_vehicle_days"].sum()
        if early["fleet_vehicle_days"].sum() > 0
        else np.nan,
        "recent_revenue_per_fleet_day": recent["revenue_krw"].sum()
        / recent["fleet_vehicle_days"].sum()
        if recent["fleet_vehicle_days"].sum() > 0
        else np.nan,
        "early_utilization_rate": early["booked_vehicle_days"].sum()
        / early["fleet_vehicle_days"].sum()
        if early["fleet_vehicle_days"].sum() > 0
        else np.nan,
        "recent_utilization_rate": recent["booked_vehicle_days"].sum()
        / recent["fleet_vehicle_days"].sum()
        if recent["fleet_vehicle_days"].sum() > 0
        else np.nan,
        "early_family_revenue_share": early["revenue_krw"].sum()
        / early["family_month_revenue_krw"].sum()
        if early["family_month_revenue_krw"].sum() > 0
        else np.nan,
        "recent_family_revenue_share": recent["revenue_krw"].sum()
        / recent["family_month_revenue_krw"].sum()
        if recent["family_month_revenue_krw"].sum() > 0
        else np.nan,
    }


def model_year_switching(monthly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    family_latest = monthly.groupby("model_family")["model_year_latest"].max().rename("newest_model_year")
    family_years = (
        monthly.groupby("model_family")["model_year_latest"]
        .nunique()
        .rename("family_model_year_count")
    )
    recent_months = sorted(monthly["month"].unique())[-RECENT_MONTHS:]
    latest_recent = monthly.merge(family_latest.reset_index(), on="model_family", how="left")
    latest_recent = latest_recent[
        (latest_recent["month"].isin(recent_months))
        & (latest_recent["model_year_latest"] == latest_recent["newest_model_year"])
    ]
    latest_share = (
        latest_recent.groupby("model_family", as_index=False)
        .agg(
            newest_recent_revenue_krw=("revenue_krw", "sum"),
            newest_recent_family_total_krw=("family_month_revenue_krw", "sum"),
        )
    )
    latest_share["newest_recent_revenue_share"] = np.where(
        latest_share["newest_recent_family_total_krw"] > 0,
        latest_share["newest_recent_revenue_krw"]
        / latest_share["newest_recent_family_total_krw"],
        np.nan,
    )
    latest_share = latest_share[["model_family", "newest_recent_revenue_share"]]

    for keys, group in monthly.groupby(["model_family", "vehicle_class", "model_year_latest"]):
        stats = window_stats(group)
        if not stats:
            continue
        row = {
            "model_family": keys[0],
            "vehicle_class": keys[1],
            "model_year_latest": keys[2],
            **stats,
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result = result.merge(family_latest.reset_index(), on="model_family", how="left")
    result = result.merge(family_years.reset_index(), on="model_family", how="left")
    result = result.merge(latest_share, on="model_family", how="left")
    result["has_newer_year_in_family"] = result["newest_model_year"] > result["model_year_latest"]
    result["year_gap_to_newest"] = result["newest_model_year"] - result["model_year_latest"]
    result["revenue_per_fleet_day_retention"] = np.where(
        result["early_revenue_per_fleet_day"] > 0,
        result["recent_revenue_per_fleet_day"] / result["early_revenue_per_fleet_day"],
        np.nan,
    )
    result["reservation_retention"] = np.where(
        result["early_reservations"] > 0,
        result["recent_reservations"] / result["early_reservations"],
        np.nan,
    )
    result["family_share_change"] = (
        result["recent_family_revenue_share"] - result["early_family_revenue_share"]
    )
    result["newer_year_pressure"] = np.where(
        result["has_newer_year_in_family"],
        result["newest_recent_revenue_share"].fillna(0),
        0,
    )
    result["hold_value_score"] = (
        result["revenue_per_fleet_day_retention"].fillna(0).clip(upper=1.5) / 1.5 * 35
        + result["reservation_retention"].fillna(0).clip(upper=1.5) / 1.5 * 20
        + result["recent_family_revenue_share"].fillna(0).clip(upper=0.5) / 0.5 * 20
        + result["recent_utilization_rate"].fillna(0).clip(upper=0.25) / 0.25 * 15
        + (1 - result["newer_year_pressure"].fillna(0).clip(upper=1)) * 10
    )
    result["cannibalization_risk_score"] = (
        (1 - result["revenue_per_fleet_day_retention"].fillna(0).clip(upper=1)) * 35
        + (1 - result["reservation_retention"].fillna(0).clip(upper=1)) * 20
        + (-result["family_share_change"].fillna(0)).clip(lower=0, upper=0.5) / 0.5 * 20
        + result["newer_year_pressure"].fillna(0).clip(upper=1) * 15
        + result["year_gap_to_newest"].clip(lower=0, upper=5) / 5 * 10
    )
    result["year_switch_signal"] = np.select(
        [
            result["has_newer_year_in_family"]
            & (result["recent_reservations"] >= 20)
            & (result["revenue_per_fleet_day_retention"] >= 0.7)
            & (result["recent_family_revenue_share"] >= 0.15),
            result["has_newer_year_in_family"]
            & (
                (result["revenue_per_fleet_day_retention"] < 0.4)
                | (result["recent_reservations"] < 5)
                | (
                    (result["family_share_change"] < -0.25)
                    & (result["newer_year_pressure"] > 0.5)
                )
            ),
        ],
        ["구형 연식도 보유 가치 있음", "신형 전환 영향 큼"],
        default="추가 관찰",
    )
    return result.sort_values("hold_value_score", ascending=False)


def family_level_signal(switching: pd.DataFrame) -> pd.DataFrame:
    old_rows = switching[switching["has_newer_year_in_family"]].copy()
    if old_rows.empty:
        return pd.DataFrame()
    family = (
        old_rows.groupby(["model_family", "vehicle_class"], as_index=False)
        .agg(
            old_year_count=("model_year_latest", "nunique"),
            newest_model_year=("newest_model_year", "max"),
            recent_old_revenue_krw=("recent_revenue_krw", "sum"),
            early_old_revenue_krw=("early_revenue_krw", "sum"),
            recent_old_reservations=("recent_reservations", "sum"),
            early_old_reservations=("early_reservations", "sum"),
            avg_recent_old_share=("recent_family_revenue_share", "mean"),
            avg_share_change=("family_share_change", "mean"),
            avg_newer_year_pressure=("newer_year_pressure", "mean"),
            avg_hold_value_score=("hold_value_score", "mean"),
            avg_cannibalization_risk_score=("cannibalization_risk_score", "mean"),
        )
    )
    family["old_revenue_retention"] = np.where(
        family["early_old_revenue_krw"] > 0,
        family["recent_old_revenue_krw"] / family["early_old_revenue_krw"],
        np.nan,
    )
    family["old_reservation_retention"] = np.where(
        family["early_old_reservations"] > 0,
        family["recent_old_reservations"] / family["early_old_reservations"],
        np.nan,
    )
    family["family_signal"] = np.select(
        [
            (family["recent_old_reservations"] >= 30)
            & (family["old_revenue_retention"] >= 0.7)
            & (family["avg_recent_old_share"] >= 0.2),
            (family["old_revenue_retention"] < 0.4)
            | (
                (family["avg_share_change"] < -0.25)
                & (family["avg_newer_year_pressure"] > 0.5)
            ),
        ],
        ["차종 단위로 구형도 유지", "차종 단위로 신형 쏠림"],
        default="차종 단위 추가 관찰",
    )
    return family.sort_values("avg_hold_value_score", ascending=False)


def format_table(df: pd.DataFrame, columns: list[str], money_cols=(), pct_cols=(), score_cols=()) -> str:
    display = df[columns].copy()
    for col in money_cols:
        if col in display:
            display[col] = display[col].map(fmt_krw)
    for col in pct_cols:
        if col in display:
            display[col] = display[col].map(fmt_pct)
    for col in score_cols:
        if col in display:
            display[col] = display[col].map(fmt_score)
    return display.to_html(index=False, escape=False)


def main() -> None:
    monthly = read_monthly_input()
    switching = model_year_switching(monthly)
    family = family_level_signal(switching)

    keep = switching[switching["year_switch_signal"].eq("구형 연식도 보유 가치 있음")].copy()
    switch = switching[switching["year_switch_signal"].eq("신형 전환 영향 큼")].copy()
    for frame in [keep, switch]:
        frame["chart_label"] = (
            frame["model_family"].astype(str)
            + " "
            + frame["model_year_latest"].astype(int).astype(str)
            + "년형"
        )
    keep_family = family[family["family_signal"].eq("차종 단위로 구형도 유지")].copy()
    switch_family = family[family["family_signal"].eq("차종 단위로 신형 쏠림")].copy()

    monthly.to_csv(OUT_DIR / "extra_monthly_family_year.csv", index=False, encoding="utf-8-sig")
    switching.to_csv(OUT_DIR / "extra_year_switching_signal.csv", index=False, encoding="utf-8-sig")
    family.to_csv(OUT_DIR / "extra_family_year_switching_signal.csv", index=False, encoding="utf-8-sig")

    total_old_rows = int(switching["has_newer_year_in_family"].sum())
    keep_recent_revenue = keep["recent_revenue_krw"].sum()
    switch_recent_revenue = switch["recent_revenue_krw"].sum()

    keep_chart = bar_chart(
        keep.sort_values("hold_value_score", ascending=False).head(8),
        "chart_label",
        "hold_value_score",
        lambda v: f"{v:.0f}점",
        "#1f8a70",
    )
    switch_chart = bar_chart(
        switch.sort_values("cannibalization_risk_score", ascending=False).head(8),
        "chart_label",
        "cannibalization_risk_score",
        lambda v: f"{v:.0f}점",
        "#c2410c",
    )
    family_keep_chart = bar_chart(
        keep_family.sort_values("avg_hold_value_score", ascending=False).head(8),
        "model_family",
        "avg_hold_value_score",
        lambda v: f"{v:.0f}점",
        "#6f42c1",
    )
    family_switch_chart = bar_chart(
        switch_family.sort_values("avg_cannibalization_risk_score", ascending=False).head(8),
        "model_family",
        "avg_cannibalization_risk_score",
        lambda v: f"{v:.0f}점",
        "#b45309",
    )

    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <title>연식 전환 기반 보유기간 대시보드</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ font-size: 26px; margin-bottom: 6px; }}
    h2 {{ font-size: 18px; margin-top: 28px; border-bottom: 1px solid #d9e2ec; padding-bottom: 6px; }}
    .note {{ color: #52606d; font-size: 13px; line-height: 1.5; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .metric {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 12px; background: #f8fafc; }}
    .label {{ color: #52606d; font-size: 12px; }}
    .value {{ font-size: 20px; font-weight: 650; margin-top: 4px; }}
    .dash {{ display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin-top: 12px; }}
    .panel {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #ffffff; }}
    .panel-title {{ font-size: 13px; font-weight: 650; margin-bottom: 10px; color: #243b53; }}
    .bar-row {{ display: grid; grid-template-columns: minmax(130px, 1.2fr) minmax(160px, 2fr) 100px; gap: 10px; align-items: center; margin: 8px 0; }}
    .bar-label {{ font-size: 12px; color: #334e68; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .bar-track {{ height: 10px; background: #edf2f7; border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ height: 100%; border-radius: 5px; }}
    .bar-value {{ font-size: 12px; color: #243b53; text-align: right; }}
    .depreciation-tool {{ border: 1px solid #d9e2ec; border-radius: 6px; padding: 14px; background: #ffffff; margin-top: 12px; }}
    .control-row {{ display: flex; flex-wrap: wrap; gap: 10px 12px; align-items: end; margin-bottom: 12px; }}
    .control-row label {{ display: flex; flex-direction: column; gap: 4px; color: #52606d; font-size: 12px; }}
    .control-row select, .control-row input {{ border: 1px solid #bcccdc; border-radius: 4px; padding: 6px 8px; min-height: 32px; font-size: 13px; color: #243b53; background: #fff; }}
    .control-row select {{ min-width: 190px; }}
    .control-row input {{ width: 92px; }}
    .control-row label:first-child select {{ width: 240px; }}
    .chart-wrap {{ width: 100%; overflow-x: auto; border: 1px solid #edf2f7; }}
    .chart-wrap svg {{ display: block; width: 100%; min-width: 760px; height: auto; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; color: #334e68; font-size: 12px; }}
    .legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
    .legend i {{ display: inline-block; width: 18px; height: 3px; border-radius: 2px; }}
    .chart-summary {{ margin-top: 10px; color: #243b53; font-size: 13px; line-height: 1.5; }}
    .insights {{ margin: 10px 0 0 18px; color: #334e68; font-size: 13px; line-height: 1.6; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; margin-top: 10px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
    th {{ background: #edf2f7; }}
    .good {{ color: #166534; font-weight: 650; }}
    .bad {{ color: #9a3412; font-weight: 650; }}
    @media (max-width: 900px) {{
      .grid, .dash {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
      .control-row {{ display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .control-row label:first-child {{ grid-column: 1 / -1; }}
      .control-row label:first-child select, .control-row input {{ width: 100%; box-sizing: border-box; }}
    }}
  </style>
</head>
<body>
  <h1>연식 전환 기반 보유기간 대시보드</h1>
  <div class="note">고정된 6년 기준이 아니라, 같은 차종 안에서 최신 연식이 존재할 때 구형 연식의 매출·예약·점유율이 유지되는지 비교했습니다.
  예: 2020년 모델과 2023년 모델이 함께 있는 차종에서 2020년 모델의 최근 매출 유지율과 최신 연식 쏠림을 봅니다.</div>
  <div class="grid">
    <div class="metric"><div class="label">신형 비교 가능한 구형 연식</div><div class="value">{total_old_rows:,}개</div></div>
    <div class="metric"><div class="label">구형도 보유 가치 있음</div><div class="value">{len(keep):,}개</div></div>
    <div class="metric"><div class="label">신형 전환 영향 큼</div><div class="value">{len(switch):,}개</div></div>
    <div class="metric"><div class="label">차종 단위 비교 가능</div><div class="value">{len(family):,}개</div></div>
  </div>
  <h2>결론</h2>
  <ul class="insights">
    <li><span class="good">구형도 보유 가치 있음</span>: 최근 3개월에도 예약과 보유일당 매출이 유지되고, 같은 차종 내 매출 점유율이 남아있는 연식입니다.</li>
    <li><span class="bad">신형 전환 영향 큼</span>: 최신 연식이 같은 차종 매출을 가져가면서 구형 연식의 최근 매출/예약 유지율 또는 점유율이 크게 떨어진 연식입니다.</li>
    <li>차종 단위 비교도 가능합니다. `model_family`로 묶어 구형 연식 전체가 유지되는 차종과 신형 쏠림 차종을 분리했습니다.</li>
    <li>감가상각 데이터가 없으므로 최종 결론은 아직 운영 수익성 신호입니다. 매입/매도 데이터가 들어오면 이 분류에 감가 비용을 더해 ROI 기준으로 바꿔야 합니다.</li>
  </ul>
  <h2>대시보드</h2>
  <div class="dash">
    <div class="panel"><div class="panel-title">구형 연식도 보유 가치 있음</div>{keep_chart}</div>
    <div class="panel"><div class="panel-title">신형 전환 영향 큰 구형 연식</div>{switch_chart}</div>
    <div class="panel"><div class="panel-title">차종 단위: 구형도 유지</div>{family_keep_chart}</div>
    <div class="panel"><div class="panel-title">차종 단위: 신형 쏠림</div>{family_switch_chart}</div>
  </div>
  {depreciation_chart_section(monthly)}
  <h2>판정 기준</h2>
  <p class="note">각 연식의 월별 매출을 기준으로 첫 3개월과 최근 3개월을 비교했습니다.
  구형 연식에 더 최신 연식이 같은 차종 안에 존재할 때, 최근 보유일당 매출 유지율, 최근 예약 유지율, 최근 차종 내 매출 점유율, 최신 연식의 최근 매출 점유율을 함께 봤습니다.
  `구형 연식도 보유 가치 있음`은 최근 예약 20건 이상, 보유일당 매출 유지율 70% 이상, 최근 차종 내 매출 점유율 15% 이상입니다.
  `신형 전환 영향 큼`은 보유일당 매출 유지율 40% 미만, 최근 예약 5건 미만, 또는 차종 내 점유율이 크게 빠지고 최신 연식 점유율이 높은 경우입니다.</p>
  <h2>구형 연식도 보유 가치 있음</h2>
  {format_table(keep.sort_values("hold_value_score", ascending=False).head(35), ["model_family", "vehicle_class", "model_year_latest", "newest_model_year", "year_gap_to_newest", "recent_reservations", "recent_revenue_krw", "recent_revenue_per_fleet_day", "revenue_per_fleet_day_retention", "recent_family_revenue_share", "newest_recent_revenue_share", "hold_value_score", "year_switch_signal"], money_cols=["recent_revenue_krw", "recent_revenue_per_fleet_day"], pct_cols=["revenue_per_fleet_day_retention", "recent_family_revenue_share", "newest_recent_revenue_share"], score_cols=["hold_value_score"])}
  <h2>신형 전환 영향 큼</h2>
  {format_table(switch.sort_values("cannibalization_risk_score", ascending=False).head(35), ["model_family", "vehicle_class", "model_year_latest", "newest_model_year", "year_gap_to_newest", "recent_reservations", "recent_revenue_krw", "recent_revenue_per_fleet_day", "revenue_per_fleet_day_retention", "family_share_change", "newest_recent_revenue_share", "cannibalization_risk_score", "year_switch_signal"], money_cols=["recent_revenue_krw", "recent_revenue_per_fleet_day"], pct_cols=["revenue_per_fleet_day_retention", "family_share_change", "newest_recent_revenue_share"], score_cols=["cannibalization_risk_score"])}
  <h2>차종 단위 비교</h2>
  {format_table(family.sort_values("avg_hold_value_score", ascending=False), ["model_family", "vehicle_class", "old_year_count", "newest_model_year", "recent_old_reservations", "recent_old_revenue_krw", "old_revenue_retention", "avg_recent_old_share", "avg_newer_year_pressure", "avg_hold_value_score", "avg_cannibalization_risk_score", "family_signal"], money_cols=["recent_old_revenue_krw"], pct_cols=["old_revenue_retention", "avg_recent_old_share", "avg_newer_year_pressure"], score_cols=["avg_hold_value_score", "avg_cannibalization_risk_score"])}
  <h2>한계</h2>
  <p class="note">분석 기간이 2025-01-01부터 2026-08-11까지라, 실제 신형 출시 직후의 전환 순간을 모두 관측하지는 못합니다.
  또한 차종 묶음은 모델명 기반 규칙 추출입니다. BMW/벤츠/포르쉐처럼 세부 모델명이 다양한 그룹은 운영자가 최종 검토해야 합니다.</p>
</body>
</html>
"""
    (OUT_DIR / "extra_analysis.html").write_text(html, encoding="utf-8")
    print(
        {
            "old_years_with_newer_comparison": total_old_rows,
            "keep_old_years": len(keep),
            "switch_to_newer_years": len(switch),
            "family_groups": len(family),
            "output": str(OUT_DIR / "extra_analysis.html"),
        }
    )


if __name__ == "__main__":
    main()
