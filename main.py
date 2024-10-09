# main.py

import asyncio
import os
from datetime import datetime
import time
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Callable
import aiohttp

from agent_config import create_agents
from user_input import UserInput
from custom_agent import CustomAgent, ReportAgent

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 환경 변수 로드
load_dotenv()

# 에러 처리를 위한 예외 클래스
class FashionRecommendationError(Exception):
    pass

class RateLimitError(Exception):
    pass

# 결과 검증 함수
def validate_results(results: List[str]) -> None:
    for result in results:
        if not result or len(result.strip()) < 100:
            raise FashionRecommendationError("분석 결과가 불충분합니다.")

async def retry_with_exponential_backoff(
    coroutine: Callable,
    max_retries: int = 5,
    base_delay: float = 1,
    max_delay: float = 60
) -> Any:
    retries = 0
    delay = base_delay

    while retries < max_retries:
        try:
            return await coroutine()
        except aiohttp.ClientResponseError as e:
            if e.status == 429:  # Rate limit error
                retries += 1
                delay = min(delay * 2, max_delay)
                logging.warning(f"Rate limit reached. Retrying in {delay:.2f} seconds... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            if "rate limit" in str(e).lower():
                retries += 1
                delay = min(delay * 2, max_delay)
                logging.warning(f"Rate limit reached. Retrying in {delay:.2f} seconds... (Attempt {retries}/{max_retries})")
                await asyncio.sleep(delay)
            else:
                raise

    raise RateLimitError("Max retries reached due to rate limiting")

async def main():
    try:
        # API 키 확인
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        # 사용자 입력 받기
        user_info = UserInput.from_console()
        current_date = datetime.now().strftime("%Y년 %m월 %d일")

        # 에이전트 생성
        user_analyst, trend_analyst, stylist, report_agent = await create_agents(api_key, user_info, current_date)

        # 태스크 정의
        tasks = [
            f"현재 날짜는 {current_date}입니다. 다음 사용자의 정보를 분석하여 체형, 스타일, 퍼스널 컬러를 파악합니다. "
            f"사용자 정보: 성별: {user_info.gender}, 키: {user_info.height}cm, 체중: {user_info.weight}kg, "
            f"예산: {user_info.budget}원, 상황: {user_info.situation}",
            
            f"현재 날짜 {current_date} 기준으로 최신 글로벌 및 한국 패션 트렌드를 조사하고 요약합니다. "
            f"사용자의 성별은 {user_info.gender}입니다. 이에 맞는 인플루언서나 연예인을 추천해주세요.",
            
            f"사용자 분석 결과와 현재 트렌드를 고려하여 개인화된 스타일과 아이템을 추천합니다. "
            f"추천 시 사용자의 예산 {user_info.budget}원 내에서 구체적인 아이템과 실제 구매 가능한 링크를 반드시 제공해야 합니다. "
            f"각 제품의 브랜드와 정확한 제품명을 명시하세요."
        ]

        # 태스크 실행 (재시도 로직 적용)
        results = await asyncio.gather(
            retry_with_exponential_backoff(lambda: user_analyst.aplan(intermediate_steps=[], input=tasks[0])),
            retry_with_exponential_backoff(lambda: trend_analyst.aplan(intermediate_steps=[], input=tasks[1])),
            retry_with_exponential_backoff(lambda: stylist.aplan(intermediate_steps=[], input=tasks[2]))
        )

        # 결과 추출
        results = [result.return_values["output"] for result in results]
        
        # 결과 검증
        validate_results(results)
        
        # 최종 보고서 생성 (재시도 로직 적용)
        final_report = await retry_with_exponential_backoff(lambda: report_agent.compile_report(*results))

        # 결과 저장
        await save_result_to_file(final_report, user_info.situation)

        logging.info("패션 분석 및 추천 보고서가 생성되었습니다.")

    except RateLimitError as e:
        logging.error(f"속도 제한 오류: {str(e)}")
        # 사용자에게 나중에 다시 시도해달라는 메시지 표시
    except FashionRecommendationError as e:
        logging.error(f"패션 추천 오류: {str(e)}")
        # 사용자에게 오류 메시지 표시 또는 재시도 로직
    except Exception as e:
        logging.error(f"예상치 못한 오류 발생: {str(e)}")
        # 일반적인 오류 처리

async def save_result_to_file(result: str, situation: str) -> None:
    current_date = datetime.now().strftime("%y%m%d")
    file_name = f"{current_date}_{situation}_패션_분석_및_추천.txt"
    
    async with asyncio.Lock():
        with open(file_name, 'w', encoding='utf-8') as file:
            file.write(result)
    
    logging.info(f"최종 보고서가 {file_name}에 저장되었습니다.")

if __name__ == "__main__":
    asyncio.run(main())