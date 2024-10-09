# custom_agent.py

from typing import List, Any, Union
from langchain.agents import AgentExecutor, create_react_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from functools import lru_cache
import re

class ImprovedOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "최종 응답:" in text:
            return AgentFinish(
                return_values={"output": text.split("최종 응답:")[-1].strip()},
                log=text,
            )
        elif "도구 사용:" in text:
            tool_name = text.split("도구 사용:")[-1].split("\n")[0].strip()
            tool_input_parts = text.split("도구 입력:")
            tool_input = tool_input_parts[-1].strip() if len(tool_input_parts) > 1 else ""
            return AgentAction(tool=tool_name, tool_input=tool_input, log=text)
        else:
            return AgentFinish(
                return_values={"output": text.strip()},
                log=text,
            )

class CustomAgent(BaseModel):
    role: str = Field(...)
    goal: str = Field(...)
    backstory: str = Field(...)
    prompt: str = Field(...)
    llm: Any = Field(...)
    tools: List[Any] = Field(...)
    agent_executor: Any = Field(None)
    memory: List[str] = Field(default_factory=list)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        super().__init__(**data)
        self._initialize_agent()

    def _initialize_agent(self):
        react_template = """
        {role}
        {goal}
        {backstory}

        {prompt}

        주의: 최종 응답에는 반드시 구체적인 브랜드명, 제품명, 구매 가능한 링크를 포함해야 합니다.
        검색 도구를 활용하여 실제 제품 정보를 찾아주세요.

        Human: {input}
        {agent_scratchpad}

        Tools available:
        {tools}

        Tool names:
        {tool_names}

        Assistant: 네, 이해했습니다. 단계별로 접근하여 요청사항을 처리하겠습니다:

        1. 먼저 주어진 정보를 분석하겠습니다.
        2. 필요한 경우 도구를 사용하여 추가 정보를 수집하겠습니다.
        3. 수집한 정보를 바탕으로 상세한 분석과 추천을 제공하겠습니다.
        4. 최종 응답에는 반드시 구체적인 브랜드명, 제품명, 구매 가능한 링크를 포함하겠습니다.

        시작하겠습니다:
        """

        prompt = PromptTemplate(
            template=react_template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "role": self.role,
                "goal": self.goal,
                "backstory": self.backstory,
                "prompt": self.prompt,
                "tools": ", ".join(tool.description for tool in self.tools),
                "tool_names": ", ".join(tool.name for tool in self.tools)
            }
        )

        output_parser = ImprovedOutputParser()
        agent = create_react_agent(self.llm, self.tools, prompt, output_parser=output_parser)
        self.agent_executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )

    @lru_cache(maxsize=100)
    def cached_tool_call(self, tool_name: str, input_text: str) -> str:
        for tool in self.tools:
            if tool.name == tool_name:
                return tool.run(input_text)
        raise ValueError(f"Tool {tool_name} not found")

    async def aplan(self, intermediate_steps: List[AgentAction], **kwargs: Any) -> Union[AgentAction, AgentFinish]:
        input_text = kwargs.get("input", "")
        self.add_to_memory(input_text)
        processed_input = self.efficient_text_processing(input_text)
        response = await self.agent_executor.ainvoke({"input": processed_input})
        output = response.get('output', str(response))
        return AgentFinish(return_values={"output": output}, log=str(response))

    def add_to_memory(self, item: str):
        if len(self.memory) >= 1000:  # 메모리 크기 제한
            self.memory.pop(0)
        self.memory.append(item)

    @staticmethod
    def efficient_text_processing(text: str) -> str:
        words = re.findall(r'\w+', text.lower())
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        return " ".join([f"{word}({freq})" for word, freq in word_freq.items()])

class ReportAgent(CustomAgent):
    async def compile_report(self, user_analysis: str, trend_analysis: str, style_recommendations: str) -> str:
        report_input = f"""
        사용자 분석: {user_analysis}
        트렌드 분석: {trend_analysis}
        스타일 추천: {style_recommendations}
        """
        result = await self.aplan(intermediate_steps=[], input=report_input)
        return result.return_values["output"]