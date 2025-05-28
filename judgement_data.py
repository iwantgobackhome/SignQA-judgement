import streamlit as st
import json
import random
import pandas as pd
from datetime import datetime

# --- 앱 설정 및 상수 ---
APP_TITLE = "대화 데이터 다중 파일 평가 도구"
EVAL_CRITERIA = [
    "1. User A의 말은 제시된 배경 상황과 하나의 자연스러운 장면으로 잘 어울리나요?",
    "2. 생성된 배경의 내용은 현실적이고 명확하게 이해하기 쉽나요?",
    "3. User B의 대답은 User A의 말과 배경 상황(background)에 논리적으로 일관성 있게 잘 연결되나요?",
    "4. User B의 대답은 문법, 어휘 사용이 적절하고 표현이 실제 대화처럼 자연스럽나요?",
    "5. 이 전체 대화(User A - 배경 - User B)는 의미 있고 자연스러워서 데이터셋에 포함할 가치가 있나요?"
]
SAMPLE_FRACTION = 0.1 # 10% 샘플링

# 사용자가 평가할 JSON 파일 목록 (실제 파일 경로로 수정해주세요)
# 예시: "data/processed_file_alpha.json"
# Streamlit Community Cloud 배포 시, 이 파일들이 GitHub 저장소 내에 함께 있어야 합니다.
# 또는, 파일 업로드 기능을 사용하여 사용자가 직접 파일을 올리도록 수정할 수도 있습니다.
DATA_FILES = {
    "how2sign_train": "data/how2sign_train/filtered_final_data.json", # 실제 파일명 또는 경로로 변경
    "how2sign_test": "data/how2sign_test/filtered_final_data.json", # 실제 파일명 또는 경로로 변경
    "how2sign_val": "data/how2sign_val/filtered_final_data.json", # 실제 파일명 또는 경로로 변경
    "openasl_data": "data/openasl_data/filtered_final_data.json"  # 실제 파일명 또는 경로로 변경
}

# --- 유틸리티 함수 ---
@st.cache_data # 데이터 로딩은 캐싱
def load_json_data(file_path):
    """지정된 경로의 JSON 파일을 로드합니다."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        st.error(f"오류: '{file_path}' 파일을 찾을 수 없습니다. 파일 경로를 확인하거나, GitHub 저장소에 파일이 올바르게 포함되었는지 확인해주세요.")
        return None
    except json.JSONDecodeError:
        st.error(f"오류: '{file_path}' 파일이 올바른 JSON 형식이 아닙니다.")
        return None
    except Exception as e:
        st.error(f"'{file_path}' 파일 로드 중 오류 발생: {e}")
        return None

def get_sampled_data(data, fraction):
    """데이터에서 지정된 비율만큼 랜덤 샘플링합니다. 최소 1개는 샘플링합니다."""
    if not data:
        return []
    num_samples = int(len(data) * fraction)
    if len(data) > 0 and num_samples == 0:
        num_samples = 1 # 최소 1개 샘플링
    if num_samples > len(data):
        num_samples = len(data) # 샘플 수가 전체 데이터 수보다 클 수 없음
    return random.sample(data, num_samples)

# --- 세션 상태 초기화 함수 ---
def initialize_session_state():
    """앱의 세션 상태 변수를 초기화합니다."""
    if "current_eval_file_key" not in st.session_state: # 현재 선택된 파일의 '키' (DATA_FILES의 키)
        st.session_state.current_eval_file_key = None
    if "sampled_data" not in st.session_state: # 현재 파일의 샘플링된 데이터
        st.session_state.sampled_data = []
    if "current_item_index" not in st.session_state: # 현재 평가 중인 아이템 인덱스
        st.session_state.current_item_index = 0
    if "evaluations_for_current_file" not in st.session_state: # 현재 파일에 대한 평가 결과 누적
        st.session_state.evaluations_for_current_file = []
    if "evaluation_of_current_file_complete" not in st.session_state:
        st.session_state.evaluation_of_current_file_complete = False
    if "all_collected_evaluations" not in st.session_state: # 모든 파일의 평가 결과를 저장 (선택적)
        st.session_state.all_collected_evaluations = {} # {file_key: [evaluations]}

def reset_evaluation_state_for_new_file(file_key):
    """새로운 파일을 평가하기 위해 관련 세션 상태를 초기화/설정합니다."""
    st.session_state.current_eval_file_key = file_key
    file_path = DATA_FILES[file_key]
    all_data_for_file = load_json_data(file_path)
    if all_data_for_file is not None:
        st.session_state.sampled_data = get_sampled_data(all_data_for_file, SAMPLE_FRACTION)
        if not st.session_state.sampled_data:
            st.warning(f"'{file_key}' 파일에서 샘플링된 데이터가 없습니다. 파일 내용을 확인해주세요.")
    else:
        st.session_state.sampled_data = [] # 로드 실패 시 빈 리스트
        
    st.session_state.current_item_index = 0
    st.session_state.evaluations_for_current_file = []
    st.session_state.evaluation_of_current_file_complete = False
    st.info(f"'{file_key}' 파일에 대한 평가를 시작합니다. 총 {len(st.session_state.sampled_data)}개의 아이템이 샘플링되었습니다.")

# --- Streamlit 앱 UI 구성 ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

initialize_session_state()

# --- 사이드바: 파일 선택 및 관리 ---
st.sidebar.header("파일 선택 및 관리")
selected_file_key_from_sidebar = st.sidebar.selectbox(
    "평가할 파일을 선택하세요:",
    options=["파일을 선택하세요..."] + list(DATA_FILES.keys()),
    index=0,
    key="sb_file_select"
)

if selected_file_key_from_sidebar != "파일을 선택하세요...":
    if st.session_state.current_eval_file_key != selected_file_key_from_sidebar:
        # 다른 파일이 선택되면, 해당 파일로 평가 상태 전환
        reset_evaluation_state_for_new_file(selected_file_key_from_sidebar)
        st.rerun() # 상태 변경 후 UI 즉시 업데이트
    # 같은 파일이 다시 선택된 경우는 현재 상태 유지 (데이터 재로딩 방지)
else: # "파일을 선택하세요..."가 선택된 경우 (초기 상태 또는 파일 선택 해제)
    if st.session_state.current_eval_file_key is not None: # 이전에 파일이 선택된 상태였다면 초기화
        st.session_state.current_eval_file_key = None
        st.session_state.sampled_data = []
        st.session_state.current_item_index = 0
        st.session_state.evaluations_for_current_file = []
        st.session_state.evaluation_of_current_file_complete = False
        # st.rerun() # 필요시 즉시 UI 업데이트

if st.sidebar.button("현재 파일 평가 초기화/다시 시작", key="btn_reset_current_file_eval"):
    if st.session_state.current_eval_file_key and st.session_state.current_eval_file_key != "파일을 선택하세요...":
        reset_evaluation_state_for_new_file(st.session_state.current_eval_file_key)
        st.success(f"'{st.session_state.current_eval_file_key}' 파일 평가가 초기화되었습니다.")
        st.rerun()
    else:
        st.sidebar.warning("초기화할 파일을 먼저 선택해주세요.")

# --- 메인 평가 영역 ---
if not st.session_state.current_eval_file_key or st.session_state.current_eval_file_key == "파일을 선택하세요...":
    st.info("👈 사이드바에서 평가할 파일을 선택해주세요.")
elif not st.session_state.sampled_data:
    st.warning(f"'{st.session_state.current_eval_file_key}' 파일에 대한 샘플링된 데이터가 없습니다. 파일이 비어있거나 로드에 실패했을 수 있습니다.")
else:
    # 현재 평가할 아이템 가져오기
    if st.session_state.current_item_index < len(st.session_state.sampled_data):
        current_item = st.session_state.sampled_data[st.session_state.current_item_index]
        item_data_id = current_item.get("data_id", f"item_idx_{st.session_state.current_item_index}")

        st.header(f"'{st.session_state.current_eval_file_key}' 파일 평가 중")
        st.subheader(f"아이템 {st.session_state.current_item_index + 1} / {len(st.session_state.sampled_data)} (ID: {item_data_id})")
        
        # 대화 내용 표시
        with st.expander("User A 발화", expanded=True):
            st.markdown(f"> {current_item.get('User A', 'N/A')}")
        
        with st.expander("배경지식 (Background)", expanded=True):
            st.markdown(f"> {current_item.get('background', 'N/A')}")
        
        with st.expander("User B 응답", expanded=True):
            st.markdown(f"> {current_item.get('User B', 'N/A')}")
        
        st.divider()
        
        # 평가 입력 폼
        with st.form(key=f"eval_form_{st.session_state.current_eval_file_key}_{item_data_id}"):
            st.subheader("평가 항목")
            current_scores = {}
            for i, criterion in enumerate(EVAL_CRITERIA):
                current_scores[criterion] = st.radio(
                    label=criterion, 
                    options=list(range(1, 6)), 
                    index=2, # 기본값 3점
                    horizontal=True, 
                    key=f"q_{i}_{item_data_id}"
                )
            
            comment = st.text_area("추가 코멘트 (선택 사항)", key=f"comment_{item_data_id}")
            
            submit_button = st.form_submit_button(label="평가 제출 및 다음 아이템")

        if submit_button:
            evaluation_entry = {
                "original_file_key": st.session_state.current_eval_file_key,
                "original_file_path": DATA_FILES[st.session_state.current_eval_file_key],
                "data_id": item_data_id,
                "sampled_item_user_a": current_item.get('User A'),
                "sampled_item_background": current_item.get('background'),
                "sampled_item_user_b": current_item.get('User B'),
                "evaluation_scores": current_scores,
                "evaluator_comment": comment,
                "evaluation_timestamp": datetime.now().isoformat()
            }
            st.session_state.evaluations_for_current_file.append(evaluation_entry)
            
            # 전체 평가 결과에도 저장 (선택적 기능)
            if st.session_state.current_eval_file_key not in st.session_state.all_collected_evaluations:
                st.session_state.all_collected_evaluations[st.session_state.current_eval_file_key] = []
            st.session_state.all_collected_evaluations[st.session_state.current_eval_file_key].append(evaluation_entry)

            st.session_state.current_item_index += 1
            
            if st.session_state.current_item_index < len(st.session_state.sampled_data):
                st.success("평가가 저장되었습니다. 다음 항목을 진행합니다.")
            else:
                st.session_state.evaluation_of_current_file_complete = True
                st.balloons()
                st.success(f"'{st.session_state.current_eval_file_key}' 파일의 모든 아이템 평가 완료!")
            st.rerun()

    elif st.session_state.evaluation_of_current_file_complete:
        st.header(f"🎉 '{st.session_state.current_eval_file_key}' 파일 평가 완료! 🎉")
        st.write(f"총 {len(st.session_state.evaluations_for_current_file)}개의 아이템에 대한 평가가 수집되었습니다.")
    else: # current_item_index >= len(sampled_data) 이지만 아직 완료 플래그가 안 선 경우 (이론상 도달 안함)
        st.info("모든 아이템 평가가 완료된 것 같습니다. 사이드바에서 다른 파일을 선택하거나 결과를 다운로드하세요.")


# --- 평가 결과 다운로드 섹션 (현재 파일에 대한 결과) ---
if st.session_state.current_eval_file_key and st.session_state.current_eval_file_key != "파일을 선택하세요...":
    if st.session_state.evaluations_for_current_file:
        st.divider()
        st.subheader(f"'{st.session_state.current_eval_file_key}' 파일 평가 결과 다운로드")
        
        # DataFrame으로 변환
        df_results = pd.DataFrame(st.session_state.evaluations_for_current_file)
        
        # evaluation_scores 딕셔너리를 별도 컬럼으로 펼치기
        # (주의: 컬럼명이 길어질 수 있음. 필요시 점수만 추출하거나 다른 방식으로 처리)
        try:
            scores_df = pd.json_normalize(df_results['evaluation_scores'])
            # 컬럼명에 접두사 추가 (예: "score_1. 맥락 적합성")
            scores_df = scores_df.rename(columns={col: f"score_{col}" for col in scores_df.columns})
            df_final_for_download = pd.concat([df_results.drop(columns=['evaluation_scores']), scores_df], axis=1)
        except Exception: # 점수 데이터가 없거나 형식이 다를 경우 원본 유지
            df_final_for_download = df_results

        # CSV 다운로드
        csv_data = df_final_for_download.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label=f"CSV 다운로드 ({len(st.session_state.evaluations_for_current_file)}개 결과)",
            data=csv_data,
            file_name=f"evaluations_{st.session_state.current_eval_file_key.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            key=f"csv_download_{st.session_state.current_eval_file_key}"
        )

        # JSON 다운로드
        json_data = json.dumps(st.session_state.evaluations_for_current_file, ensure_ascii=False, indent=2)
        st.download_button(
            label=f"JSON 다운로드 ({len(st.session_state.evaluations_for_current_file)}개 결과)",
            data=json_data,
            file_name=f"evaluations_{st.session_state.current_eval_file_key.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            key=f"json_download_{st.session_state.current_eval_file_key}"
        )

# --- 앱 정보 ---
st.sidebar.divider()
st.sidebar.markdown("---")
st.sidebar.info(f"{APP_TITLE}\n\nStreamlit을 활용한 평가 도구입니다.")
