from langchain_groq import ChatGroq
from typing import Dict, Any

# 각 에이전트별 모델 설정
AGENT_MODELS = {
    "user_analyst": "llama-3.2-90b-text-preview",
    "trend_analyst": "llama-3.2-90b-text-preview",
    "stylist": "llama-3.2-90b-text-preview",
    "report_agent": "llama-3.2-90b-text-preview"
}

async def initialize_llm(api_key: str, agent_name: str) -> Any:
    """
    Groq LLM을 비동기적으로 초기화합니다.
    :param api_key: Groq API 키
    :param agent_name: 에이전트 이름
    :return: 초기화된 LLM 객체
    """
    model_name = AGENT_MODELS.get(agent_name, "llama-3.2-90b-text-preview")  # 기본값 설정
    return ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0,  # 생성 텍스트의 창의성 조절
        max_tokens=None,  # 생성할 최대 토큰 수
        model_kwargs={"top_p": 0.9}  # top_p를 model_kwargs로 이동
    )

def get_model_name(agent_name: str) -> str:
    """
    에이전트 이름에 해당하는 모델 이름을 반환합니다.
    """
    return AGENT_MODELS.get(agent_name, "llama-3.2-90b-text-preview")