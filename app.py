import streamlit as st
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any, List
import os
from dataclasses import dataclass
import logging
from pathlib import Path
from dotenv import load_dotenv

from agent_config import create_agents
from user_input import UserInput

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class UserProfile:
    gender: str
    height: float
    weight: float
    budget: int
    situation: str
    style_preferences: Optional[str] = None
    tpo: Optional[str] = None
    
    @property
    def bmi(self) -> float:
        return self.weight / ((self.height / 100) ** 2)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'budget': self.budget,
            'situation': self.situation,
            'tpo': self.tpo,
            'image_paths': []
        }

class ImageProcessor:
    def __init__(self):
        self.uploaded_images: List[str] = []
        self.temp_dir = "temp"
        self._ensure_temp_directory()
    
    def _ensure_temp_directory(self) -> None:
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
            logging.info(f"Created temporary directory: {self.temp_dir}")
    
    def _clean_temp_directory(self) -> None:
        for file_path in Path(self.temp_dir).glob("*"):
            try:
                file_path.unlink()
            except Exception as e:
                logging.warning(f"Failed to delete {file_path}: {e}")
    
    def process_upload(self, uploaded_file) -> bool:
        if uploaded_file is None:
            return False
            
        try:
            safe_filename = "".join(c for c in uploaded_file.name if c.isalnum() or c in "._-")
            file_path = os.path.join(self.temp_dir, safe_filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            self.uploaded_images.append(file_path)
            logging.info(f"Successfully saved uploaded file: {file_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to process uploaded file: {e}")
            st.error(f"이미지 업로드 중 오류가 발생했습니다: {str(e)}")
            return False
    
    def get_uploaded_images(self) -> List[str]:
        return self.uploaded_images
    
    def clear_uploads(self) -> None:
        self._clean_temp_directory()
        self.uploaded_images.clear()

class StyleAdvisor:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.current_date = datetime.now().strftime("%Y년 %m월 %d일")

    async def generate_recommendations(self, user_profile: UserProfile) -> Dict[str, Any]:
        try:
            # UserInput 객체 생성
            user_info = UserInput(**user_profile.to_dict())

            # 에이전트 생성
            user_analyst, trend_analyst, stylist, report_agent = await create_agents(
                self.api_key,
                user_info,
                self.current_date
            )
            
            # 분석 태스크 정의
            tasks = [
                user_analyst.aplan(
                    intermediate_steps=[],
                    input=f"현재 날짜는 {self.current_date}입니다. "
                          f"다음 사용자의 정보를 분석하여 체형, 스타일, 퍼스널 컬러를 파악합니다. "
                          f"사용자 정보: 성별: {user_profile.gender}, "
                          f"키: {user_profile.height}cm, 체중: {user_profile.weight}kg, "
                          f"BMI: {user_profile.bmi:.1f}, "
                          f"예산: {user_profile.budget}원, TPO: {user_profile.tpo}, "
                          f"상황: {user_profile.situation}"
                ),
                trend_analyst.aplan(
                    intermediate_steps=[],
                    input=f"현재 날짜 {self.current_date} 기준으로 "
                          f"최신 글로벌 및 한국 패션 트렌드를 조사하고 요약합니다. "
                          f"사용자의 성별은 {user_profile.gender}입니다."
                ),
                stylist.aplan(
                    intermediate_steps=[],
                    input=f"사용자 정보와 현재 트렌드를 고려하여 개인화된 스타일과 아이템을 추천합니다. "
                          f"예산 {user_profile.budget}원 내에서 구체적인 아이템과 "
                          f"실제 구매 가능한 링크를 제공해야 합니다. "
                          f"TPO: {user_profile.tpo}, 상황: {user_profile.situation}"
                )
            ]
            
            # 병렬로 분석 실행
            results = await asyncio.gather(*tasks)
            analyses = [result.return_values["output"] for result in results]
            
            # 최종 보고서 생성
            final_report = await report_agent.compile_report(*analyses)
            
            return {
                "user_analysis": analyses[0],
                "trend_analysis": analyses[1],
                "style_recommendations": analyses[2],
                "final_report": final_report
            }
            
        except Exception as e:
            logging.error(f"Style recommendation generation failed: {str(e)}")
            raise

class StreamlitApp:
    def __init__(self):
        st.set_page_config(
            page_title="AI 패션 스타일리스트",
            page_icon="👔",
            layout="wide"
        )
        self.image_processor = ImageProcessor()
        self.initialize_session_state()
        
    @staticmethod
    def initialize_session_state():
        if 'current_step' not in st.session_state:
            st.session_state.current_step = 0
        if 'user_profile' not in st.session_state:
            st.session_state.user_profile = None
        if 'recommendations' not in st.session_state:
            st.session_state.recommendations = None
    
    def render_user_input_form(self):
        st.title("AI 패션 스타일리스트 🎨")
        
        with st.form("user_info_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                gender = st.selectbox("성별", ["남성", "여성"])
                height = st.number_input("키 (cm)", min_value=140.0, max_value=200.0, value=170.0)
                weight = st.number_input("체중 (kg)", min_value=30.0, max_value=150.0, value=65.0)
            
            with col2:
                budget = st.number_input(
                    "예산 (원)", 
                    min_value=50000, 
                    max_value=10000000, 
                    value=300000,
                    step=50000
                )
                tpo = st.text_input(
                    "TPO (Time, Place, Occasion)",
                    placeholder="예: 주말, 회사, 결혼식 등"
                )
                situation = st.text_input(
                    "구체적인 상황을 설명해주세요",
                    placeholder="예: 주말 데이트, 회사 면접, 결혼식 하객 등"
                )
            
            style_preferences = st.text_area(
                "선호하는 스타일이나 특별한 요구사항이 있다면 알려주세요 (선택사항)",
                placeholder="예: 모던한 스타일 선호, 화려한 색상 피하고 싶음 등"
            )
            
            uploaded_file = st.file_uploader(
                "참고할 이미지를 업로드해주세요 (선택사항)", 
                type=["jpg", "jpeg", "png"]
            )
            
            submit_button = st.form_submit_button("스타일 분석 시작", use_container_width=True)
            
            if submit_button:
                if not situation or not tpo:
                    st.error("TPO와 구체적인 상황을 모두 입력해주세요.")
                    return
                
                if uploaded_file:
                    if not self.image_processor.process_upload(uploaded_file):
                        st.error("이미지 업로드에 실패했습니다. 다시 시도해주세요.")
                        return
                
                st.session_state.user_profile = UserProfile(
                    gender=gender,
                    height=height,
                    weight=weight,
                    budget=budget,
                    situation=situation,
                    style_preferences=style_preferences,
                    tpo=tpo
                )
                st.session_state.current_step = 1
    
    async def generate_and_display_recommendations(self):
        try:
            api_key = os.getenv('GROQ_API_KEY')
            if not api_key:
                st.error("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")
                st.session_state.current_step = 0
                return
                
            style_advisor = StyleAdvisor(api_key)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner('AI 스타일리스트가 당신을 위한 최적의 스타일을 분석 중입니다...'):
                try:
                    status_text.text("체형 분석 중...")
                    progress_bar.progress(25)
                    
                    recommendations = await style_advisor.generate_recommendations(
                        st.session_state.user_profile
                    )
                    progress_bar.progress(50)
                    status_text.text("트렌드 분석 중...")
                    
                    if recommendations:
                        progress_bar.progress(75)
                        status_text.text("스타일 추천 생성 중...")
                        st.session_state.recommendations = recommendations
                        
                        progress_bar.progress(100)
                        status_text.text("분석이 완료되었습니다!")
                        time.sleep(1)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.session_state.current_step = 2
                        st.rerun()
                    else:
                        raise ValueError("추천 결과가 생성되지 않았습니다.")
                        
                except Exception as e:
                    st.error(f"스타일 분석 중 오류가 발생했습니다: {str(e)}")
                    logging.error(f"Recommendation generation error: {str(e)}")
                    st.session_state.current_step = 0
                    
        except Exception as e:
            st.error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            logging.error(f"Unexpected error: {str(e)}")
            st.session_state.current_step = 0
    
    def display_results(self):
        try:
            if not st.session_state.recommendations or not st.session_state.user_profile:
                st.error("분석 결과를 찾을 수 없습니다. 다시 시도해주세요.")
                st.session_state.current_step = 0
                return

            recommendations = st.session_state.recommendations
            user_profile = st.session_state.user_profile
            
            st.title("당신을 위한 맞춤 스타일 제안 📋")
            
            # 사용자 정보 표시
            with st.expander("입력하신 정보", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("### 기본 정보")
                    st.write(f"- 성별: {user_profile.gender}")
                    st.write(f"- 키: {user_profile.height}cm")
                    st.write(f"- 체중: {user_profile.weight}kg")
                    st.write(f"- BMI: {user_profile.bmi:.1f}")
                with col2:
                    st.write("### 스타일링 정보")
                    st.write(f"- 예산: {user_profile.budget:,}원")
                    st.write(f"- TPO: {user_profile.tpo}")
                    st.write(f"- 상황: {user_profile.situation}")
                    if user_profile.style_preferences:
                        st.write(f"- 스타일 선호도: {user_profile.style_preferences}")

            # 분석 결과 탭
            tab1, tab2, tab3, tab4 = st.tabs(["👤 체형 분석", "📈 트렌드 분석", "👔 스타일 추천", "📋 종합 보고서"])
            
            with tab1:
                st.markdown("### 체형 분석")
                if "user_analysis" in recommendations:
                    st.markdown(recommendations["user_analysis"])
                else:
                    st.warning("체형 분석 결과를 찾을 수 없습니다.")
            
            with tab2:
                st.markdown("### 트렌드 분석")
                if "trend_analysis" in recommendations:
                    st.markdown(recommendations["trend_analysis"])
                else:
                    st.warning("트렌드 분석 결과를 찾을 수 없습니다.")
            
            with tab3:
                st.markdown("### 스타일 추천")
                if "style_recommendations" in recommendations:
                    st.markdown(recommendations["style_recommendations"])
                else:
                    st.warning("스타일 추천 결과를 찾을 수 없습니다.")
            
            with tab4:
                st.markdown("### 종합 보고서")
                if "final_report" in recommendations:
                    st.markdown(recommendations["final_report"])
                else:
                    st.warning("종합 보고서를 찾을 수 없습니다.")
            
            # 새로운 분석 시작 버튼
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("새로운 분석 시작", key="restart_button", use_container_width=True):
                    self.cleanup()
                    st.session_state.current_step = 0
                    st.session_state.user_profile = None
                    st.session_state.recommendations = None
                    st.rerun()

        except Exception as e:
            st.error(f"결과 표시 중 오류가 발생했습니다: {str(e)}")
            logging.error(f"Display results error: {str(e)}")
            st.session_state.current_step = 0

    def cleanup(self):
        """세션 종료 시 정리 작업을 수행합니다."""
        self.image_processor.clear_uploads()
        
    def run(self):
        """애플리케이션을 실행합니다."""
        try:
            if st.session_state.current_step == 0:
                self.render_user_input_form()
            elif st.session_state.current_step == 1:
                asyncio.run(self.generate_and_display_recommendations())
            elif st.session_state.current_step == 2:
                self.display_results()
        except Exception as e:
            st.error(f"오류가 발생했습니다: {str(e)}")
            logging.error(f"Application error: {str(e)}")
        finally:
            # 세션이 재시작되거나 종료될 때 정리
            if st.session_state.current_step == 0:
                self.cleanup()

def main():
    """메인 함수"""
    try:
        logging.info("애플리케이션 시작")
        app = StreamlitApp()
        app.run()
    except Exception as e:
        st.error(f"애플리케이션 실행 중 오류가 발생했습니다: {str(e)}")
        logging.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()