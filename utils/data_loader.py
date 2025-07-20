"""
Data Loader
데이터 로딩 및 전처리
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


class DataLoader:
    """
    데이터 로딩 및 전처리 클래스
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        # 디렉토리 생성
        self._create_directories()
    
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        for directory in [self.data_dir, self.raw_dir, self.processed_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
    
    def load_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        시장 데이터 로딩
        
        Args:
            start_date: 시작일 (YYYY-MM-DD)
            end_date: 종료일 (YYYY-MM-DD)
            
        Returns:
            시장 데이터 DataFrame
        """
        # 실제 구현에서는 API나 데이터베이스에서 데이터를 가져옴
        # 여기서는 샘플 데이터를 생성
        
        # 날짜 범위 생성
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start, end, freq='D')
        
        # 샘플 종목 리스트
        sample_stocks = [
            '005930', '000660', '035420', '051910', '006400',  # 삼성전자, SK하이닉스, NAVER, LG화학, 삼성SDI
            '035720', '207940', '068270', '323410', '373220',  # 카카오, 삼성바이오로직스, 셀트리온, 카카오뱅크, LG에너지솔루션
            '005380', '051900', '006980', '035250', '017670'   # 현대차, LG생활건강, 우성사료, 강원랜드, SK텔레콤
        ]
        
        # 샘플 데이터 생성
        data = []
        for date in date_range:
            for stock in sample_stocks:
                # 랜덤 가격 데이터 생성
                base_price = np.random.uniform(50000, 200000)
                price_change = np.random.normal(0, 0.02)  # 2% 표준편차
                current_price = base_price * (1 + price_change)
                
                # OHLCV 데이터
                open_price = current_price * (1 + np.random.normal(0, 0.005))
                high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.01)))
                low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.01)))
                volume = np.random.randint(1000000, 10000000)
                
                # 재무 지표 (간단한 랜덤 값)
                pe_ratio = np.random.uniform(5, 50)
                pb_ratio = np.random.uniform(0.5, 5)
                roe = np.random.uniform(0.05, 0.25)
                roa = np.random.uniform(0.02, 0.15)
                debt_equity_ratio = np.random.uniform(0.1, 1.5)
                current_ratio = np.random.uniform(1.0, 3.0)
                market_cap = current_price * np.random.randint(1000000, 100000000)
                
                data.append({
                    'date': date,
                    'symbol': stock,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': current_price,
                    'volume': volume,
                    'pe_ratio': pe_ratio,
                    'pb_ratio': pb_ratio,
                    'roe': roe,
                    'roa': roa,
                    'debt_equity_ratio': debt_equity_ratio,
                    'current_ratio': current_ratio,
                    'market_cap': market_cap,
                    'returns': price_change,
                    'sector': self._get_sector(stock)
                })
        
        df = pd.DataFrame(data)
        df.set_index(['date', 'symbol'], inplace=True)
        
        return df
    
    def _get_sector(self, stock_code: str) -> str:
        """종목 코드로 섹터 반환"""
        # 간단한 섹터 매핑
        sector_mapping = {
            '005930': '전자', '000660': '전자', '035420': 'IT서비스',
            '051910': '화학', '006400': '전자', '035720': 'IT서비스',
            '207940': '바이오', '068270': '바이오', '323410': '금융',
            '373220': '화학', '005380': '자동차', '051900': '화학',
            '006980': '식품', '035250': '서비스', '017670': '통신'
        }
        return sector_mapping.get(stock_code, '기타')
    
    def load_financial_data(self, symbols: List[str], 
                           start_date: str, end_date: str) -> pd.DataFrame:
        """
        재무 데이터 로딩
        
        Args:
            symbols: 종목 코드 리스트
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            재무 데이터 DataFrame
        """
        # 실제 구현에서는 재무제표 API에서 데이터를 가져옴
        # 여기서는 샘플 데이터를 생성
        
        data = []
        for symbol in symbols:
            # 분기별 재무 데이터 생성
            quarters = pd.date_range(start_date, end_date, freq='Q')
            
            for quarter in quarters:
                # 랜덤 재무 지표 생성
                revenue = np.random.uniform(1000000000000, 10000000000000)  # 1조~10조
                net_income = revenue * np.random.uniform(0.05, 0.15)  # 5~15% 마진
                total_assets = revenue * np.random.uniform(1.5, 3.0)
                total_equity = total_assets * np.random.uniform(0.3, 0.7)
                total_debt = total_assets - total_equity
                
                # 성장률 계산
                revenue_growth = np.random.uniform(-0.1, 0.3)
                earnings_growth = np.random.uniform(-0.2, 0.4)
                
                data.append({
                    'date': quarter,
                    'symbol': symbol,
                    'revenue': revenue,
                    'net_income': net_income,
                    'total_assets': total_assets,
                    'total_equity': total_equity,
                    'total_debt': total_debt,
                    'revenue_growth': revenue_growth,
                    'earnings_growth': earnings_growth,
                    'roe': net_income / total_equity,
                    'roa': net_income / total_assets,
                    'debt_ratio': total_debt / total_assets
                })
        
        df = pd.DataFrame(data)
        df.set_index(['date', 'symbol'], inplace=True)
        
        return df
    
    def load_economic_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        경제 지표 데이터 로딩
        
        Args:
            start_date: 시작일
            end_date: 종료일
            
        Returns:
            경제 지표 DataFrame
        """
        # 실제 구현에서는 경제 지표 API에서 데이터를 가져옴
        # 여기서는 샘플 데이터를 생성
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = pd.date_range(start, end, freq='M')
        
        data = []
        for date in date_range:
            # 랜덤 경제 지표 생성
            kospi_return = np.random.normal(0.01, 0.05)  # 월 수익률
            interest_rate = np.random.uniform(1.0, 3.0)  # 금리
            exchange_rate = np.random.uniform(1100, 1300)  # 환율
            inflation_rate = np.random.uniform(1.0, 4.0)  # 인플레이션
            
            data.append({
                'date': date,
                'kospi_return': kospi_return,
                'interest_rate': interest_rate,
                'exchange_rate': exchange_rate,
                'inflation_rate': inflation_rate,
                'market_volatility': abs(kospi_return) * 2
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        return df
    
    def preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 전처리
        
        Args:
            raw_data: 원본 데이터
            
        Returns:
            전처리된 데이터
        """
        processed_data = raw_data.copy()
        
        # 결측값 처리
        processed_data = processed_data.fillna(method='ffill')
        processed_data = processed_data.fillna(method='bfill')
        
        # 이상값 처리 (3시그마 규칙)
        numeric_columns = processed_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            mean = processed_data[col].mean()
            std = processed_data[col].std()
            
            # 3시그마 범위를 벗어나는 값들을 NaN으로 처리
            processed_data[col] = processed_data[col].apply(
                lambda x: np.nan if abs(x - mean) > 3 * std else x
            )
        
        # 결측값을 다시 처리
        processed_data = processed_data.fillna(method='ffill')
        processed_data = processed_data.fillna(method='bfill')
        
        # 추가 지표 계산
        if 'close' in processed_data.columns and 'volume' in processed_data.columns:
            processed_data['market_cap'] = processed_data['close'] * processed_data['volume']
        
        if 'pe_ratio' in processed_data.columns and 'pb_ratio' in processed_data.columns:
            processed_data['peg_ratio'] = processed_data['pe_ratio'] / (processed_data['roe'] * 100)
        
        return processed_data
    
    def save_processed_data(self, data: pd.DataFrame, filename: str):
        """
        전처리된 데이터 저장
        
        Args:
            data: 저장할 데이터
            filename: 파일명
        """
        filepath = os.path.join(self.processed_dir, filename)
        data.to_csv(filepath)
        print(f"데이터가 저장되었습니다: {filepath}")
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """
        전처리된 데이터 로딩
        
        Args:
            filename: 파일명
            
        Returns:
            로딩된 데이터
        """
        filepath = os.path.join(self.processed_dir, filename)
        
        if os.path.exists(filepath):
            data = pd.read_csv(filepath, index_col=[0, 1] if ',' in data.columns[0] else 0)
            return data
        else:
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {filepath}")
    
    def get_stock_universe(self, market_cap_min: float = 100000000000, 
                          market_cap_max: float = float('inf')) -> List[str]:
        """
        투자 대상 종목 리스트 반환
        
        Args:
            market_cap_min: 최소 시가총액
            market_cap_max: 최대 시가총액
            
        Returns:
            종목 코드 리스트
        """
        # 실제 구현에서는 시가총액 기준으로 종목을 필터링
        # 여기서는 샘플 종목 리스트 반환
        
        sample_stocks = [
            '005930', '000660', '035420', '051910', '006400',
            '035720', '207940', '068270', '323410', '373220',
            '005380', '051900', '006980', '035250', '017670'
        ]
        
        return sample_stocks 