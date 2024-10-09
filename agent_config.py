from typing import Tuple
from custom_agent import CustomAgent, ReportAgent
from config import initialize_llm
from langchain.tools import Tool
from langchain.tools.ddg_search.tool import DuckDuckGoSearchRun
from langchain_community.tools.youtube.search import YouTubeSearchTool
from langchain_community.tools import ArxivQueryRun
from prompts import USER_ANALYST_PROMPT, TREND_ANALYST_PROMPT, STYLIST_PROMPT, REPORT_AGENT_PROMPT
from user_input import UserInput

async def create_agents(api_key: str, user_info: UserInput, current_date: str) -> Tuple[CustomAgent, CustomAgent, CustomAgent, ReportAgent]:
    ddg_search = DuckDuckGoSearchRun()
    youtube_search = YouTubeSearchTool()
    arxiv = ArxivQueryRun()

    tools = [
        Tool(
            name="DuckDuckGo Search",
            func=ddg_search.run,
            description="최신 패션 트렌드, 브랜드 정보, 스타일 팁 등을 검색합니다."
        ),
        Tool(
            name="YouTube Search",
            func=youtube_search.run,
            description="패션 쇼, 스타일 튜토리얼, 트렌드 분석 영상 등을 검색합니다."
        ),
        Tool(
            name="Arxiv",
            func=arxiv.run,
            description="패션 관련 연구, 트렌드 분석, 소비자 행동 등에 대한 학술 정보를 검색합니다."
        )
    ]

    user_info_dict = user_info.__dict__
    user_info_dict['current_date'] = current_date

    user_analyst_llm = await initialize_llm(api_key, "user_analyst")
    trend_analyst_llm = await initialize_llm(api_key, "trend_analyst")
    stylist_llm = await initialize_llm(api_key, "stylist")
    report_agent_llm = await initialize_llm(api_key, "report_agent")

    user_analyst = CustomAgent(
        role="사용자 분석가",
        goal="사용자의 체형, 스타일, 퍼스널 컬러를 정확히 분석합니다.",
        backstory="당신은 패션 업계에서 20년 이상의 경력을 가진 전문 이미지 컨설턴트입니다.",
        llm=user_analyst_llm,
        tools=tools,
        prompt=USER_ANALYST_PROMPT.format(**user_info_dict)
    )

    trend_analyst = CustomAgent(
        role="트렌드 분석가",
        goal="최신 패션 트렌드를 분석하고 스타일리스트에게 정보를 제공합니다.",
        backstory="당신은 세계적인 패션 매거진의 수석 에디터로, 글로벌 패션 트렌드를 분석하는 전문가입니다.",
        llm=trend_analyst_llm,
        tools=tools,
        prompt=TREND_ANALYST_PROMPT.format(**user_info_dict)
    )

    stylist = CustomAgent(
        role="AI 스타일리스트",
        goal="사용자에게 최적화된 패션 스타일과 아이템을 추천합니다.",
        backstory="당신은 셀러브리티들의 스타일링을 담당하는 최고의 패션 스타일리스트입니다.",
        llm=stylist_llm,
        tools=tools,
        prompt=STYLIST_PROMPT.format(**user_info_dict)
    )

    report_agent = ReportAgent(
        role="리포트 작성자",
        goal="다른 에이전트들의 분석 결과를 종합하여 가독성 높은 패션 분석 및 추천 보고서를 작성합니다.",
        backstory="당신은 패션 업계의 전문 리포트 작성자로, 복잡한 정보를 명확하고 실용적인 보고서로 정리하는 능력이 뛰어납니다.",
        llm=report_agent_llm,
        tools=tools,
        prompt=REPORT_AGENT_PROMPT.format(**user_info_dict)
    )

    return user_analyst, trend_analyst, stylist, report_agent