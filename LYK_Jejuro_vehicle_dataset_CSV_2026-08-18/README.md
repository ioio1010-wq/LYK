# LYK Jejuro vehicle dataset — 2026-08-18

- Source: ERP MySQL read-only, company.id=7 (제주로렌트카), queried 2026-08-18
- Related Linear issue: GNO-59 [LYK] 외부 공유용 데이터셋 v1 준비 (related existing issue reused as delivery context)
- Workbook: `LYK_Jejuro_vehicle_dataset_2026-08-18.xlsx`
- Privacy: excludes vehicle plate numbers and all customer/driver PII.
- Value classes: observed, derived, unavailable. No estimated values.

## Populated
- Vehicle master: 3,438 ERP vehicle records (912 current non-deleted; 2,526 historical soft-deleted records).
- Current model fleet snapshot: 141 models, 912 current non-deleted records at 2026-08-18 06:17:45 UTC.
- Maintenance: 15,318 non-deleted events and 3,438 vehicle summaries.
- Vehicle reservation metrics: 80 vehicles / 80 valid reservations using `assigned_vehicle_id` direct join only.
- Model reservation metrics: 300 models; monthly extract 10,713 model-month rows (workbook only).

## Unavailable / intentionally blank
Acquisition date, purchase price, verified sale/disposal date, sale price, holding period, fleet insurance cost, historical daily fleet counts, true utilization rate, and accounting-recognized revenue.

`vehicle.registered_at` is not used as acquisition date. `vehicle.deleted_at` is not used as sale/disposal date. Model `release_price` is not used as vehicle purchase price. Customer insurance fee is not used as fleet insurance cost.
