"""
Base Agent Template for Quantitative Investment Strategies
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, TypedDict
import pandas as pd
import numpy as np

from langchain_core.runnables import Runnable, RunnableSequence, RunnableLambda
from langchain_core.pydantic_v1 import BaseModel, Field
from pydantic import ConfigDict
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import os


class AgentInput(BaseModel):
    """에이전트 입력 모델"""
    tickers: List[str] = Field(description="분석할 종목 리스트")
    start_date: str = Field(description="시작일 (YYYY-MM-DD)")
    end_date: str = Field(description="종료일 (YYYY-MM-DD)")
    top_n: int = Field(default=10, description="상위 몇 개 종목을 선택할지")


class AgentOutput(BaseModel):
    """에이전트 출력 모델"""
    screened_data: Any = Field(description="스크리닝된 종목 데이터")
    top_stocks: Any = Field(description="상위 종목들")
    llm_explanation: str = Field(description="LLM 설명")
    stock_count: int = Field(description="스크리닝된 종목 수")
    agent_name: str = Field(description="에이전트 이름")
    
    class Config:
        arbitrary_types_allowed = True


class BaseAgent(ABC):
    """
    모든 퀀트 투자 에이전트의 기본 클래스 (RunnableSequence 기반)
    """
    
    def __init__(self, name: str, description: str, llm=None):
        self.name = name
        self.description = description
        self.parameters = {}
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        
        # LLM 설정
        api_key = os.getenv('OPENAI_API_KEY')
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4o", api_key=api_key)
        
        # 체인 구성: 데이터 수집 → 스크리닝 → 점수 계산 → 설명 생성
        self.chain = self._build_chain()
    
    def _build_chain(self) -> RunnableSequence:
        """
        에이전트 실행 체인 구성: enrich_data → screen → score → explain
        """
        enrich_step = RunnableLambda(self._enrich_data_step)
        screen_step = RunnableLambda(self._screen_stocks_step) 
        score_step = RunnableLambda(self._calculate_scores_step)
        explain_step = RunnableLambda(self._explain_with_llm_step)
        
        return RunnableSequence(
            first=enrich_step,
            middle=[screen_step, score_step],
            last=explain_step
        )
    
    def _enrich_data_step(self, agent_input: AgentInput) -> Dict[str, Any]:
        """데이터 수집 단계"""
        enriched_data = self.enrich_data(
            agent_input.tickers, 
            agent_input.start_date, 
            agent_input.end_date
        )
        return {
            "enriched_data": enriched_data,
            "tickers": agent_input.tickers,
            "start_date": agent_input.start_date,
            "end_date": agent_input.end_date,
            "top_n": agent_input.top_n
        }
    
    def _screen_stocks_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """스크리닝 단계"""
        screened_data = self.screen_stocks(data["enriched_data"])
        data["screened_data"] = screened_data
        return data
    
    def _calculate_scores_step(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """점수 계산 단계"""
        if len(data["screened_data"]) > 0:
            data["screened_data"]['score'] = data["screened_data"].apply(
                self.calculate_score, axis=1
            )
            top_stocks = data["screened_data"].sort_values(
                'score', ascending=False
            ).head(data["top_n"])
        else:
            top_stocks = pd.DataFrame()
        
        data["top_stocks"] = top_stocks
        return data
    
    def _explain_with_llm_step(self, data: Dict[str, Any]) -> AgentOutput:
        """LLM 설명 생성 단계"""
        llm_explanation = self.explain_topN_with_llm(
            data["screened_data"], 
            data["top_n"],
            data["start_date"],
            data["end_date"]
        )
        
        return AgentOutput(
            screened_data=data["screened_data"],
            top_stocks=data["top_stocks"], 
            llm_explanation=llm_explanation,
            stock_count=len(data["screened_data"]),
            agent_name=self.name
        )
    
    def invoke(self, agent_input: AgentInput) -> AgentOutput:
        """체인 실행"""
        return self.chain.invoke(agent_input)
    
    # 추상 메서드들 (하위 클래스에서 구현)
    @abstractmethod
    def enrich_data(self, tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """데이터 수집 메서드"""
        pass
    
    @abstractmethod
    def screen_stocks(self, data: pd.DataFrame) -> pd.DataFrame:
        """주식 스크리닝을 수행하는 추상 메서드"""
        pass
    
    @abstractmethod
    def calculate_score(self, stock_data: pd.Series) -> float:
        """개별 종목에 대한 점수를 계산하는 추상 메서드"""
        pass
    
    @abstractmethod
    def explain_topN_with_llm(self, screened_data: pd.DataFrame, top_n: int, 
                             start_date: str, end_date: str) -> str:
        """LLM을 통한 상위 종목 설명 메서드"""
        pass
        
    def get_parameters(self) -> Dict[str, Any]:
        """에이전트의 파라미터를 반환"""
        return self.parameters
    
    def set_parameters(self, parameters: Dict[str, Any]):
        """에이전트의 파라미터를 설정"""
        self.parameters.update(parameters) 