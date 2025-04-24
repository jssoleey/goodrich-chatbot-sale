import streamlit as st
from llm_sale import get_chatbot_response, get_script_response, get_kakao_response
import re
import os, json
from datetime import datetime, timedelta, timezone

# ----------------- 마크다운 자동 정리 함수 -------------------
def format_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    formatted_lines = []
    indent_next = False

    for line in lines:
        line = line.strip()
        if not line:
            formatted_lines.append("")
            indent_next = False
            continue

        if re.match(r"^(▶️|✅|📌|❗|📝|📍)\s*[^:：]+[:：]?", line):
            title = re.sub(r"[:：]\s*$", "", line.strip())
            formatted_lines.append(f"**{title}**\n")
            indent_next = False
            continue

        if re.match(r"^[-•]\s*\*\*.*\*\*", line):
            formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            indent_next = True
            continue

        if re.match(r"^[-•]\s*", line):
            if indent_next:
                formatted_lines.append("    " + re.sub(r"^[-•]\s*", "- ", line))
            else:
                formatted_lines.append(re.sub(r"^[-•]\s*", "- ", line))
            continue

        formatted_lines.append(line)
        indent_next = False

    return "\n".join(formatted_lines).strip() + "\n"

# ----------------- 페이지 설정 -------------------
st.set_page_config( 
    page_title="스마트 컨설팅 매니저",
    page_icon="https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/logo.png?raw=true"
)

# 이미지 URL
top_image_url = "https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/top_box.png?raw=true"

# 최상단에 이미지 출력
st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:20px;">
        <img src="{top_image_url}" alt="Top Banner" style="width:100%; max-width:1000px;">
    </div>
    """,
    unsafe_allow_html=True
)

logo_url = "https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/logo.png?raw=true"
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: -10px;">
        <img src="{logo_url}" alt="logo" width="50">
        <h2 style="margin: 0;">스마트 컨설팅 매니저</h2>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("입력하신 고객 정보에 따라 상담 스크립트를 만들어 드립니다!")
st.caption("정보가 구체적일수록 좋은 스크립트가 나와요.")
st.caption("스크립트 생성 이후 추가적인 대화를 통해 AI에게 상황을 현재 알려주세요!")
st.caption("대화가 끝나면 '카카오톡 문자 생성하기' 기능을 활용해보세요 😊")

st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)
st.markdown('<p class="small-text">모든 답변은 참고용으로 활용해주세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)

# ----------------- CSS -------------------
st.markdown(
    """
    <style>
    .small-text {
        font-size: 12px;
        color: gray;
        line-height: 1.3;
        margin-top: 4px;
        margin-bottom: 4px;
    }
    .user-message {
        background-color: #e6e6e6;
        color: black;
        padding: 15px;
        border-radius: 30px;
        max-width: 80%;
        text-align: left;
        word-wrap: break-word;
    }
    .ai-message {
        background-color: #ffffff;
        color: black;
        padding: 10px;
        border-radius: 10px;
        max-width: 70%;
        text-align: left;
        word-wrap: break-word;
    }
    .message-container {
        display: flex;
        align-items: flex-start;
        margin-bottom: 10px;
    }
    .message-container.user {
        justify-content: flex-end;
    }
    .message-container.ai {
        justify-content: flex-start;
    }
    .avatar {
        width: 50px;
        height: 50px;
        border-radius: 0%;
        margin: 0 10px;
    }
    .input-box {
        background: #ff9c01;
        padding: 10px;
        border-radius: 0px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .input-line {
        background: #ff9c01;
        padding: 1px;
        border-radius: 0px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);
        margin-bottom: 10px;
    }
    .custom-button > button {
        background-color: #ff6b6b;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        border: none;
    }
    /* 사이드바 전체 여백 조정 */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: -50px;    /* 상단 여백 */
        padding-bottom: 0px;  /* 하단 여백 */
        padding-left: 5px;
        padding-right: 5px;
    }

    /* 사이드바 내부 요소 간격 줄이기 */
    .block-container div[data-testid="stVerticalBlock"] {
        margin-top: -5px;
        margin-bottom: -5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- 세션 상태 초기화 -------------------
if 'page' not in st.session_state:
    st.session_state.page = "login"

if 'page' not in st.session_state:
    st.session_state.page = "input"  # 시작은 고객 정보 입력 화면

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# ----------------- 메시지 표시 함수 -------------------
def display_message(role, content, avatar_url):
    if role == "user":
        alignment = "user"
        message_class = "user-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        message_html = f'<div class="{message_class}">{content}</div>'
        display_html = f"""
        <div class="message-container {alignment}">
            {message_html}
            {avatar_html}
        </div>
        """
        st.markdown(display_html, unsafe_allow_html=True)
    else:
        alignment = "ai"
        message_class = "ai-message"
        avatar_html = f'<img src="{avatar_url}" class="avatar">'
        display_html = f"""
        <div class="message-container {alignment}">
            {avatar_html}
            <div class="{message_class}">
        """
        st.markdown(display_html, unsafe_allow_html=True)
        st.markdown(format_markdown(content), unsafe_allow_html=False)
        st.markdown("</div></div>", unsafe_allow_html=True)

# ----------------- 로그인 -------------------
if st.session_state.page == "login":
    name = st.text_input(label = "ID", placeholder="이름(홍길동)")
    emp_id = st.text_input(label = "Password", placeholder="휴대폰 끝번호 네 자리(0000)")
            
    col1, col2, col3 = st.columns([1, 1, 1])   # 비율을 조정해서 가운데로

    with col2:
        if st.button("로그인", use_container_width=True):
            if name and emp_id:
                st.session_state['user_folder'] = f"{name}_{emp_id}"
                st.session_state['user_name'] = name   # ✅ 상담원 이름 따로 저장
                st.session_state.page = "input"
                st.experimental_rerun()
            else:
                st.warning("이름과 생일을 모두 입력해 주세요.")


# ----------------- 고객 정보 입력 화면 -------------------
if st.session_state.page == "input":

    # 현재 날짜와 시간 표시
    KST = timezone(timedelta(hours=9))
    now_korea = datetime.now(KST).strftime("%Y년 %m월 %d일")
    st.sidebar.markdown(
        f"<span style='font-size:18px;'>📅 <b>{now_korea}</b></span>",
        unsafe_allow_html=True
    )
    
    # 최상단 인삿말 + 화이팅 멘트
    user_name = st.session_state['user_folder'].split('_')[0]
    st.sidebar.title(f"😊 {user_name}님, 반갑습니다!")
    st.sidebar.markdown("오늘도 멋진 상담 화이팅입니다! 💪")

    # 구분선
    st.sidebar.markdown(
        "<hr style='margin-top:14px; margin-bottom:28px;'>",
        unsafe_allow_html=True
    )

    # 사용자 폴더 경로
    user_path = f"/data/history/{st.session_state['user_folder']}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    history_files = os.listdir(user_path)

    if history_files:
        # 🔍 검색창 추가
        search_keyword = st.sidebar.text_input("🔎 고객명으로 검색", placeholder="고객명 입력 후 ENTER", key="search_input")        

        # 파일명 필터링 (대소문자 무시)
        filtered_files = [f for f in history_files if search_keyword.lower() in f.lower()]
        
        selected_chat = st.sidebar.selectbox("📂 저장된 대화 기록", filtered_files)
        
        col1, col2 = st.sidebar.columns(2)
        

        with col1:
            if st.button("불러오기", use_container_width=True):
                with open(f"{user_path}/{selected_chat}", "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        st.session_state['script_context'] = ""
                        st.session_state.message_list = loaded_data
                        st.session_state['customer_name'] = "고객명미입력"
                    elif isinstance(loaded_data, dict):
                        st.session_state['script_context'] = loaded_data.get("script_context", "")
                        st.session_state.message_list = loaded_data.get("message_list", [])
                        st.session_state['customer_name'] = loaded_data.get("customer_name", selected_chat.split('_')[0])
                    else:
                        st.error("❌ 불러온 파일 형식이 잘못되었습니다.")
                        st.stop()

                st.session_state['current_file'] = selected_chat
                st.session_state.page = "chatbot"
                st.experimental_rerun()

        with col2:
            if st.button("🗑️ 삭제하기", use_container_width=True):
                file_path = f"{user_path}/{selected_chat}"
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        st.sidebar.success(f"{selected_chat} 삭제 완료!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ 삭제 중 오류가 발생했습니다: {e}")
                else:
                    st.sidebar.warning("이미 삭제된 파일입니다.")
        # ✅ 검색 결과 없을 때만 메시지 출력
        if not filtered_files and search_keyword:
            st.sidebar.markdown(
                "<div style='padding:6px; background-color:#f0f0f0; border-radius:5px;'>🔍 검색 결과가 없습니다.</div>",
                unsafe_allow_html=True
            )
            
    else:
        st.sidebar.info("저장된 대화가 없습니다.")

    # 구분선
    st.sidebar.markdown(
        "<hr style='margin-top:14px; margin-bottom:28px;'>",
        unsafe_allow_html=True
    )
    
    # 새로운 고객 정보 입력하기 버튼       
    if st.sidebar.button("🆕 새로운 고객 정보 입력하기", use_container_width=True):
        st.session_state.page = "input"
        st.session_state.message_list = []
        st.session_state.script_context = ""
        st.session_state.kakao_text = ""
        st.session_state['current_file'] = ""  # 👉 덮어쓰기 방지
        st.experimental_rerun()    

    # 최하단 로그아웃 버튼
    if st.sidebar.button("로그아웃", use_container_width=True):
        st.session_state.page = "login"
        st.session_state.message_list = []
        st.experimental_rerun()
    
    st.markdown(
        "<h4 style='margin-bottom: 20px;'>👤 고객 정보를 입력해 주세요</h4>",
        unsafe_allow_html=True
    )
    name = st.text_input("고객 이름")
    age_group = st.selectbox("고객 연령대", ["연령대를 선택하세요", "20대", "30대", "40대", "50대", "60대 초반", "60대 후반 이상"])
    gender = st.selectbox("고객 성별", ["성별을 선택하세요", "남성", "여성"])
    insurance_status = st.text_input(
        label="기존 보험 상태",
        placeholder="예: 태아보험 상담 신청, 10년 전에 가입한 암보험과 실손보험이 있음, 보장 내용은 잘 모름"
    )
    interest = st.text_input(
        "고객 관심 보험",
        placeholder="예: 태아보험, 간병보험"
    )

    reaction = st.text_input(
        "고객 반응",
        placeholder="예: 보험료를 저렴하게 가입하고 싶음, 최근 병원 진료 후 필요성을 느껴 상담 신청"
    )

    etc = st.text_input(
        "기타 상황",
        placeholder="예: 가족력(부친 고혈압) 있고, 갱신형 보험료 인상에 대한 걱정이 있음"
    )

    col1, col2, col3 = st.columns([1, 2, 1])   # 비율을 조정해서 가운데로

    with col2:
        if st.button("🚀 상담 스크립트 생성하기", use_container_width=True):
            if (
                name 
                and insurance_status 
                and interest 
                and reaction 
                and etc 
                and age_group != "연령대를 선택하세요"
                and gender != "성별을 선택하세요"
            ):
                # 고객 이름 세션에 저장
                st.session_state['customer_name'] = name
                
                customer_info = f"""
                    - 고객 이름: {name}
                    - 고객 연령대: {age_group}
                    - 고객 성별: {gender}
                    - 기존 보험 상태: {insurance_status}
                    - 고객 관심 보험: {interest}
                    - 고객 반응: {reaction}
                    - 기타 상황: {etc}
                    """
                with st.spinner("상담 스크립트를 생성 중입니다..."):
                    ai_response = get_script_response(customer_info)
                    script_text = "".join(ai_response)

                    # 👉 스크립트 context 저장
                    st.session_state['script_context'] = script_text

                    # 상담 스크립트를 첫 메시지로 저장
                    st.session_state.message_list = []
                    st.session_state.message_list.append({"role": "ai", "content": script_text})

                st.session_state.page = "chatbot"
                st.experimental_rerun() 
            else:
                st.warning("모든 항목을 입력해 주세요.")
                
# ----------------- 챗봇 화면 -------------------
elif st.session_state.page == "chatbot":
        
    # 현재 날짜와 시간 표시
    KST = timezone(timedelta(hours=9))
    now_korea = datetime.now(KST).strftime("%Y년 %m월 %d일")
    st.sidebar.markdown(
        f"<span style='font-size:18px;'>📅 <b>{now_korea}</b></span>",
        unsafe_allow_html=True
    )
    
    # 최상단 인삿말 + 화이팅 멘트
    user_name = st.session_state['user_folder'].split('_')[0]
    st.sidebar.title(f"😊 {user_name}님, 반갑습니다!")
    st.sidebar.markdown("오늘도 멋진 상담 화이팅입니다! 💪")

    # 구분선
    st.sidebar.markdown(
        "<hr style='margin-top:14px; margin-bottom:28px;'>",
        unsafe_allow_html=True
    )

    # 사용자 폴더 경로
    user_path = f"/data/history/{st.session_state['user_folder']}"
    if not os.path.exists(user_path):
        os.makedirs(user_path)

    history_files = os.listdir(user_path)

    if history_files:
        # 🔍 검색창 추가
        search_keyword = st.sidebar.text_input("🔎 고객명으로 검색", placeholder="고객명 입력 후 ENTER", key="search_input")        

        # 파일명 필터링 (대소문자 무시)
        filtered_files = [f for f in history_files if search_keyword.lower() in f.lower()]
        
        selected_chat = st.sidebar.selectbox("📂 저장된 대화 기록", filtered_files)
        
        col1, col2 = st.sidebar.columns(2)
        

        with col1:
            if st.button("불러오기", use_container_width=True):
                with open(f"{user_path}/{selected_chat}", "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        st.session_state['script_context'] = ""
                        st.session_state.message_list = loaded_data
                        st.session_state['customer_name'] = "고객명미입력"
                    elif isinstance(loaded_data, dict):
                        st.session_state['script_context'] = loaded_data.get("script_context", "")
                        st.session_state.message_list = loaded_data.get("message_list", [])
                        st.session_state['customer_name'] = loaded_data.get("customer_name", selected_chat.split('_')[0])
                    else:
                        st.error("❌ 불러온 파일 형식이 잘못되었습니다.")
                        st.stop()

                st.session_state['current_file'] = selected_chat
                st.session_state.page = "chatbot"
                st.experimental_rerun()

        with col2:
            if st.button("🗑️ 삭제하기", use_container_width=True):
                file_path = f"{user_path}/{selected_chat}"
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        st.sidebar.success(f"{selected_chat} 삭제 완료!")
                        st.experimental_rerun()
                    except Exception as e:
                        st.sidebar.error(f"❌ 삭제 중 오류가 발생했습니다: {e}")
                else:
                    st.sidebar.warning("이미 삭제된 파일입니다.")
        # ✅ 검색 결과 없을 때만 메시지 출력
        if not filtered_files and search_keyword:
            st.sidebar.markdown(
                "<div style='padding:6px; background-color:#f0f0f0; border-radius:5px;'>🔍 검색 결과가 없습니다.</div>",
                unsafe_allow_html=True
            )
            
    else:
        st.sidebar.info("저장된 대화가 없습니다.")

    # 구분선
    st.sidebar.markdown(
        "<hr style='margin-top:14px; margin-bottom:28px;'>",
        unsafe_allow_html=True
    )
    
    # 새로운 고객 정보 입력하기 버튼       
    if st.sidebar.button("🆕 새로운 고객 정보 입력하기", use_container_width=True):
        st.session_state.page = "input"
        st.session_state.message_list = []
        st.session_state.script_context = ""
        st.session_state.kakao_text = ""
        st.session_state['current_file'] = ""  # 👉 덮어쓰기 방지
        st.experimental_rerun()    

    # 최하단 로그아웃 버튼
    if st.sidebar.button("로그아웃", use_container_width=True):
        st.session_state.page = "login"
        st.session_state.message_list = []
        st.experimental_rerun()
        
    user_avatar = "https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/user_avatar.png?raw=true"
    ai_avatar = "https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/ai_avatar.png?raw=true"
        
    messages = st.session_state.get("message_list", [])

    if isinstance(messages, list):
        for message in messages:
            if isinstance(message, dict) and "role" in message and "content" in message:
                role = message["role"]
                content = message["content"]
                avatar = user_avatar if role == "user" else ai_avatar
                display_message(role, content, avatar)
            else:
                st.warning("⚠️ 불러온 메시지 형식이 잘못되었습니다.")
    else:
        st.error("❌ 메시지 리스트가 손상되었습니다. 다시 불러와 주세요.")

    if user_question := st.chat_input("보험 상담 관련 추가 질문을 입력해 주세요."):
        st.session_state.message_list.append({"role": "user", "content": user_question})
        display_message("user", user_question, user_avatar)

        with st.spinner("답변을 준비 중입니다..."):
            ai_response = get_chatbot_response(user_question, st.session_state['script_context'])
            formatted_response = format_markdown("".join(ai_response))
            st.session_state.message_list.append({"role": "ai", "content": formatted_response})
            display_message("ai", formatted_response, ai_avatar)

    # 👉 버튼 영역: 두 개의 버튼을 나란히 배치
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("💬 카카오톡 발송용 문자 생성하기", use_container_width=True):
            with st.spinner("카카오톡 문자를 생성 중입니다..."):
                kakao_message = get_kakao_response(
                    script_context = st.session_state['script_context'],
                    message_list = st.session_state['message_list']
                )
                st.session_state['kakao_text'] = "".join(kakao_message)
                                
    with col2:
        if st.button("💾 대화 저장하기", use_container_width=True):
            user_path = f"/data/history/{st.session_state['user_folder']}"
            if not os.path.exists(user_path):
                os.makedirs(user_path)
            if st.session_state.message_list:
                # 1️⃣ 고객 이름 확보
                customer_name = st.session_state.get('customer_name', '고객명미입력')

                # 2️⃣ 기존 파일명 여부 확인
                if st.session_state.get('current_file'):
                    # 기존 파일명에서 고객 이름 유지, 시간만 갱신
                    KST = timezone(timedelta(hours=9))
                    new_filename = f"{customer_name}_{datetime.now(KST).strftime('%y%m%d-%H%M%S')}.json"
                    
                    # 기존 파일 삭제 (덮어쓰기 효과)
                    old_file = f"{user_path}/{st.session_state['current_file']}"
                    if os.path.exists(old_file):
                        os.remove(old_file)
                else:
                    # 새로운 저장이라면
                    KST = timezone(timedelta(hours=9))
                    new_filename = f"{customer_name}_{datetime.now(KST).strftime('%y%m%d-%H%M%S')}.json"

                # 3️⃣ 데이터 저장
                data_to_save = {
                    "customer_name": customer_name,
                    "script_context": st.session_state.get('script_context', ''),
                    "message_list": st.session_state.message_list
                }

                with open(f"{user_path}/{new_filename}", "w", encoding="utf-8") as f:
                    json.dump(data_to_save, f, ensure_ascii=False, indent=4)

                # 4️⃣ 파일명 업데이트
                st.session_state['current_file'] = new_filename

                st.success(f"대화가 저장되었습니다! ({new_filename})")
            else:
                st.warning("저장할 대화가 없습니다.")
    
    # 👉 생성된 카카오톡 문자 출력 (있을 때만 표시)
    if st.session_state.get('kakao_text'):
        st.markdown("### 📩 카카오톡 발송용 문자")
        st.text_area("아래 내용을 복사해 사용하세요.", value=st.session_state['kakao_text'], height=400)
        
# 이미지 URL
bottom_image_url = "https://github.com/jssoleey/goodrich-chatbot-sale/blob/main/image/bottom_box.png?raw=true"

# 최하단에 이미지 출력
st.caption("")

st.markdown(
    f"""
    <div style="text-align:center; margin-bottom:20px;">
        <img src="{bottom_image_url}" alt="Top Banner" style="width:100%; max-width:1000px;">
    </div>
    """,
    unsafe_allow_html=True
)
