# Analysis5 Package

`extra_analysis.html`과 관련 코드, 문서, 재생성용 집계 데이터를 묶은 폴더입니다.

## 구성

- `html/extra_analysis.html`: 연식 전환 기반 보유기간 대시보드와 차종별 매출·감가 시뮬레이션 HTML
- `code/build_extra_dashboard.py`: `extra_analysis.html` 재생성 스크립트
- `code/holding_period_analysis.py`: 원본 예약/차량 데이터를 정리하고 보유기간 분석을 만드는 기반 코드
- `docs/summary.md`: 분석 요약 문서
- `data/extra_monthly_family_year.csv`: 대시보드 재생성에 필요한 월별 차종·연식 집계 데이터
- `data/extra_year_switching_signal.csv`: 연식 전환 신호 결과
- `data/extra_family_year_switching_signal.csv`: 차종 단위 연식 전환 신호 결과

## 재생성 참고

원본 예약 데이터가 없는 환경에서는 `build_extra_dashboard.py`가 `data/extra_monthly_family_year.csv`와 같은 집계 파일을 사용하도록 조정되어 있습니다.
