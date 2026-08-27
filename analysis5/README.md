# LYK 분석 결과 한눈에 보기

이 저장소의 `analysis1` ~ `analysis5`는 제주로 렌터카 데이터를 서로 다른 의사결정 질문에 맞춰 정리한 분석 산출물입니다.

| 폴더 | 분석 주제 | 핵심 질문 | 주요 결과물 |
|---|---|---|---|
| `analysis1` | 유저/예약 퍼널 분석 | 유저가 방문부터 예약, 결제까지 어디서 이탈하는가? 채널/캠페인/언어/보험 선택에 따라 전환율이 어떻게 다른가? | `analysis1_summary.html`, `overall_cumulative_funnel.csv`, `channel_user_funnel_summary.csv`, `reservation_channel_summary.csv`, `segment_diagnostics/segment_diagnostics_report.html` |
| `analysis2` | 차량별 예약 성과 분석 | 어떤 차량/차종이 예약 수요와 요금합계 기준으로 좋았고, 어떤 차량은 주의가 필요한가? | `index.html`, `vehicle_metrics.csv`, `vehicle_monthly_metrics.csv`, `expansion_candidates.csv`, `caution_candidates.csv`, `car_type_metrics.csv` |
| `analysis3` | 품절/가동률/증차 기회 분석 | 보유 차량 대비 수요가 초과되는 모델은 무엇이며, 추가 1~2대 투입 시 어느 모델의 기회가 큰가? | `index.html`, `summary.md`, `daily_model_capacity.csv`, `model_capacity_summary.csv`, `stockout_periods.csv`, `incremental_vehicle_profit_proxy.csv`, `model_profit_proxy.csv` |
| `analysis4` | 적정 보유기간 신호 분석 | 차량 연식이 오래될수록 매출과 가동률이 어떻게 변하며, 6년 이상 차량 중 계속 보유할 만한 모델은 무엇인가? | `index.html`, `summary.md`, `age_curve_summary.csv`, `model_holding_period_signal.csv`, `class_holding_period_recommendation.csv`, `old_vehicle_customer_profile.csv`, `family_year_comparison.csv` |
| `analysis5` | 연식 전환 기반 보유기간 대시보드 패키지 | 같은 차종 안에서 신형이 들어왔을 때 구형 연식의 매출/예약 점유율이 유지되는가, 아니면 신형으로 수요가 이동하는가? | `html/extra_analysis.html`, `docs/summary.md`, `data/extra_year_switching_signal.csv`, `data/extra_family_year_switching_signal.csv`, `code/build_extra_dashboard.py` |

## 분석별 요약

### `analysis1` - 퍼널/세그먼트 진단

Mixpanel 기반 유저 퍼널을 정리한 분석입니다. `identity_level = reservation_id`처럼 customer_id 없이 reservation_id만 있는 1,176개 미식별 예약 단위는 유저 분석에서 제외하고, 총 67,638개 canonical user를 기준으로 방문, 식별, 검색, 차량 클릭, 예약 시작, 보험 선택, 결제 시도, 결제 완료 단계를 봅니다.

핵심 지표는 예약 생성 유저 9,685명, 결제 완료 유저 2,963명, 유저 기준 결제 완료율 4.4%입니다. 추가 진단에서는 source/medium/campaign/language별 퍼널 도달률, 보험 선택률, 보험 미선택 이탈, 가격대별 결제율 차이를 통계적으로 비교했습니다.

먼저 볼 파일:

- `analysis1/analysis1_summary.html`: 전체 퍼널 요약 리포트
- `analysis1/segment_diagnostics/segment_diagnostics_report.html`: 채널/캠페인/언어별 세그먼트 진단 리포트
- `analysis1/segment_diagnostics/notion_step_conversion_table.csv`: Notion 등에 붙여넣기 쉬운 단계별 전환표

### `analysis2` - 차량별 예약 성과

`Mixpanel_partner`의 2026년 1월~7월 월별 Parquet 파일 7개에서 `reservation_created` 이벤트를 예약 1건으로 보고, 차량별 예약 건수와 요금합계를 비교한 분석입니다. 총 11,726,603개 이벤트에서 15,186건의 예약을 추출했고, 차량명 매핑률은 100%입니다.

핵심 목적은 실제 구매 확정이 아니라 "확대 검토 후보"와 "주의 후보"를 선별하는 것입니다. 총액 1위는 `26년 팰리세이드4륜(휘)`, 예약 건수 1위는 `25년 레이(휘)`, 차종 기준 요금합계 1위는 SUV입니다.

먼저 볼 파일:

- `analysis2/index.html`: 차량별 예약 성과 대시보드
- `analysis2/expansion_candidates.csv`: 확대 검토 후보
- `analysis2/caution_candidates.csv`: 최근 성과 하락 등 주의 후보
- `analysis2/vehicle_metrics.csv`: 차량별 전체 지표

### `analysis3` - 품절/가동률/증차 기회

예약, 사전예약, 차량 마스터, 정비 요약 데이터를 결합해 모델-일자 단위의 보유대수, 예약 시간, 최대 동시 예약 수를 계산한 분석입니다. 품절일은 모델-일자별 최대 동시 예약 수가 해당 일자의 보유 차량 수 이상인 날로 정의했습니다.

분석 기간은 2025-01-01 ~ 2026-08-11이며, 유효 예약은 35,972건입니다. 전체 평균 가동률은 16.5%, 품절 모델-일자는 1,545일입니다. 추가 1대/2대 투입 시 흡수 가능한 미전환 사전예약을 바탕으로 증차 매출 및 이익 프록시도 계산했습니다.

먼저 볼 파일:

- `analysis3/index.html`: 품절/가동률/증차 기회 리포트
- `analysis3/summary.md`: 핵심 수치 요약
- `analysis3/model_capacity_summary.csv`: 모델별 가동률과 품절 지표
- `analysis3/incremental_vehicle_profit_proxy.csv`: 추가 1대/2대 증차 기회 프록시

### `analysis4` - 적정 보유기간 신호

차량 연식과 보유 차량 일수를 기준으로 차령별 매출 유지력과 가동률을 본 분석입니다. 실제 매입일, 매입가, 매각일, 매각가가 없어서 감가상각 기반의 확정 ROI가 아니라 운영 데이터 기반 보유기간 신호로 해석해야 합니다.

분석 기간은 2025-01-01 ~ 2026-08-11이며, 유효 예약은 35,944건입니다. 6년 이상 연식 매출은 112,362,903원, 전체 매출 비중은 2.2%입니다. 6년 이상에서도 수요가 유지되는 모델과, 차급별로 교체 검토가 시작될 수 있는 차령 구간을 구분했습니다.

먼저 볼 파일:

- `analysis4/index.html`: 적정 보유기간 신호 리포트
- `analysis4/summary.md`: 핵심 수치 요약
- `analysis4/age_curve_summary.csv`: 차령 구간별 매출/가동률
- `analysis4/model_holding_period_signal.csv`: 모델별 장기 보유 신호
- `analysis4/class_holding_period_recommendation.csv`: 차급별 교체 검토 구간

### `analysis5` - 연식 전환 기반 보유기간 대시보드

`analysis4`의 보유기간 분석을 더 세분화해, 같은 차종 안에서 구형 연식과 최신 연식이 함께 있을 때 수요가 어디로 이동하는지 본 대시보드 패키지입니다. 예를 들어 같은 `투싼`, `팰리세이드`, `K5` 계열 안에서 구형 연식의 최근 매출 유지율, 예약 유지율, 차종 내 매출 점유율, 최신 연식 쏠림 정도를 비교합니다.

핵심 분류는 `구형 연식도 보유 가치 있음`, `신형 전환 영향 큼`, `추가 관찰`입니다. 대시보드에는 차종별 매출선과 감가 가정을 입력해 보는 간단한 매출-감가 시뮬레이션도 포함되어 있습니다.

먼저 볼 파일:

- `analysis5/html/extra_analysis.html`: 연식 전환 기반 보유기간 대시보드
- `analysis5/docs/summary.md`: 분석 요약
- `analysis5/data/extra_year_switching_signal.csv`: 연식별 전환 신호
- `analysis5/data/extra_family_year_switching_signal.csv`: 차종 단위 전환 신호

## 공통 해석 주의사항

- `analysis2`의 매출은 실제 결제 승인액이 아니라 Mixpanel 예약 생성 이벤트의 `total_amount` 기준 요금합계입니다.
- `analysis3`~`analysis5`는 실제 매입가, 매각가, 감가상각, 보험/정비/운영비 전체가 반영된 회계상 순이익 분석이 아닙니다.
- 차량별 개별 VIN/차량 ID 단위 수익성이 아니라, 대부분 모델/차급/연식 단위의 운영 프록시입니다.
- 구매, 증차, 매각 결정을 확정하려면 매입가, 매각 예상가, 실제 정비비, 보험료, 취소/환불, 차량별 실보유대수 정보를 추가로 결합해야 합니다.
