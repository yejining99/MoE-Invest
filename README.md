# MOE-INVEST (Multi-Agent Optimization Engine for Investment)

MOE-INVEST는 여러 퀀트 투자 에이전트의 의견을 종합하여 최적의 투자 결정을 내리는 멀티 에이전트 시스템입니다.

## 🎯 프로젝트 개요

이 시스템은 다음과 같은 투자 대가들의 전략을 에이전트로 구현합니다:

- **벤자민 그레이엄 (Graham)**: 가치투자 전략
- **피오트로스키 (Piotroski)**: F-Score 기반 재무 건전성 평가
- **조엘 그린블랫 (Greenblatt)**: Magic Formula 전략
- **칼라일 (Carlisle)**: 모멘텀 기반 투자
- **리처드 드리하우스 (Driehaus)**: 성장 투자 전략

## 🏗️ 시스템 아키텍처

```
MOE-INVEST/
│
├── data/                        # OHLCV, 재무지표, 지표 계산용
│   ├── raw/                     # 원본 CSV, API 데이터 등
│   └── processed/               # 전처리된 스크리닝 대상 종목 리스트
│
├── agents/                      # 각 퀀트 대가 Agent 정의
│   ├── base_agent.py            # Agent 템플릿 클래스
│   ├── graham.py
│   ├── piotroski.py
│   ├── greenblatt.py
│   ├── carlisle.py
│   └── driehaus.py
│
├── meta_agent/                 # 마스터 에이전트 + CoT reasoning
│   ├── cot_reasoner.py          # CoT 기반 판단자 (LLM wrapper)
│   └── prompt_templates.py      # LLM 프롬프트 구조
│
├── portfolio/                   # 포트폴리오 구성 및 리밸런싱
│   ├── constructor.py
│   └── evaluator.py             # Sharpe, MDD, Turnover 등
│
├── backtest/                    # 실험 루틴
│   ├── run_backtest.py
│   └── result_logger.py
│
├── notebooks/                   # 실험용 노트북 or 시각화
│   └── 01_agent_outputs_demo.ipynb
│
├── utils/                       # 데이터 로딩, 재무 지표 계산 등
│   ├── data_loader.py
│   └── finance_utils.py
│
└── config.yaml                  # 실험 config (기간, 자산군, 기준 등)
```

## 🚀 주요 기능

### 1. 멀티 에이전트 투자 시스템
- 5개의 서로 다른 투자 철학을 가진 에이전트
- 각 에이전트의 독립적인 스크리닝 및 점수 계산
- 에이전트 간 합의도 기반 최종 결정

### 2. Chain of Thought (CoT) 추론
- LLM을 활용한 투자 결정 추론
- 각 에이전트의 투자 철학을 고려한 종합 분석
- 시장 상황과 리스크를 고려한 최종 선별

### 3. 포트폴리오 최적화
- 리스크 조정 수익률 최대화
- 다각화 효과 극대화
- 정기적인 리밸런싱

### 4. 백테스트 및 성과 평가
- 다양한 성과 지표 계산 (Sharpe, MDD, VaR 등)
- 에이전트별 성과 비교
- 시각화 및 보고서 생성

## 📦 설치 및 실행

### 1. 의존성 설치
```bash
pip install pandas numpy matplotlib seaborn scipy pyyaml
```

### 2. 설정 파일 수정
`config.yaml` 파일에서 백테스트 기간, 에이전트 파라미터 등을 설정합니다.

### 3. 백테스트 실행
```python
from backtest.run_backtest import BacktestRunner
from backtest.result_logger import ResultLogger

# 백테스트 실행
runner = BacktestRunner(config)
results = runner.run_backtest("2020-01-01", "2023-12-31")

# 결과 로깅
logger = ResultLogger()
logger.log_backtest_results(results, "experiment_1")
```

### 4. 노트북 실행
```bash
jupyter notebook notebooks/01_agent_outputs_demo.ipynb
```

## 🔧 설정 옵션

### 에이전트별 주요 파라미터

#### Graham Agent
- `max_pe`: 최대 P/E 비율 (기본값: 15)
- `max_pb`: 최대 P/B 비율 (기본값: 1.5)
- `min_current_ratio`: 최소 유동비율 (기본값: 2.0)

#### Piotroski Agent
- `min_f_score`: 최소 F-Score (기본값: 7)
- `min_market_cap`: 최소 시가총액 (기본값: 10억원)

#### Greenblatt Agent
- `min_roic`: 최소 ROIC (기본값: 5%)
- `max_ev_ebit`: 최대 EV/EBIT (기본값: 50)

#### Carlisle Agent
- `momentum_period`: 모멘텀 계산 기간 (기본값: 252일)
- `min_momentum`: 최소 모멘텀 (기본값: 5%)

#### Driehaus Agent
- `min_revenue_growth`: 최소 매출 성장률 (기본값: 15%)
- `min_earnings_growth`: 최소 이익 성장률 (기본값: 20%)

## 📊 성과 지표

시스템은 다음과 같은 성과 지표를 계산합니다:

- **수익률 지표**: 총 수익률, 연간화 수익률
- **리스크 지표**: 변동성, 최대 낙폭, VaR, CVaR
- **리스크 조정 지표**: 샤프 비율, 소르티노 비율, 칼마 비율
- **포트폴리오 지표**: 다각화 점수, 리밸런싱 비용
