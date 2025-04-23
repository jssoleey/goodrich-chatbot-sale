import streamlit as st
from llm_sale import get_chatbot_response, get_script_response, get_kakao_response
import re

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

# ----------------- 페이지 설정 및 스타일 -------------------
st.set_page_config(
    page_title="스마트 컨설팅 매니저",
    page_icon="https://img-insur.richnco.co.kr/goodrichmall/common/favi_goodrichmall.ico"
)

logo_url = "https://img-insur.richnco.co.kr/goodrichmall/common/favi_goodrichmall.ico"
st.markdown(
    f"""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: -10px;">
        <img src="{logo_url}" alt="logo" width="50">
        <h1 style="margin: 0;">스마트 컨설팅 매니저</h1>
    </div>
    """,
    unsafe_allow_html=True
)
st.caption("입력하신 고객 정보에 따라 상담 스크립트를 만들어 드립니다!")
st.caption("정보가 구체적일수록 좋은 스크립트가 나와요!")

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
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin: 0 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="small-text">모든 답변은 참고용으로 활용해주세요.</p>', unsafe_allow_html=True)
st.markdown('<p class="small-text"> </p>', unsafe_allow_html=True)

# ----------------- 세션 상태 초기화 -------------------
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

# ----------------- 고객 정보 입력 화면 -------------------
if st.session_state.page == "input":
    st.markdown(
        "<h4 style='margin-bottom: 20px;'>👤 고객 정보를 입력해 주세요</h4>",
        unsafe_allow_html=True
    )

    name = st.text_input("고객 이름")
    age_group = st.selectbox("고객 연령대", ["연령대를 선택하세요", "20대", "30대", "40대", "50대", "60대 초반", "60대 후반 이상"])
    gender = st.selectbox("고객 성별", ["성별을 선택하세요", "남성", "여성"])
    insurance_status = st.text_input(
        label="기존 보험 상태",
        placeholder="예: 태아보험 상담 신청, 예전에 보험 가입, 최근 내용은 기억 못함"
    )
    interest = st.text_input(
        "고객 관심 보험",
        placeholder="예: 치매보험, 간병보험"
    )

    reaction = st.text_input(
        "고객 반응",
        placeholder="예: 보험료를 저렴하게 가입하고 싶음, 기억은 안 나지만 시간 괜찮다고 함"
    )

    etc = st.text_input(
        "기타 상황",
        placeholder="예: 기존 보험료가 다소 부담스러웠음, 배우자 병간호 경험으로 장기 보장 필요성 느낌"
    )

    if st.button("🚀 상담 스크립트 생성"):
        if (
            name 
            and insurance_status 
            and interest 
            and reaction 
            and etc 
            and age_group != "연령대를 선택하세요"
            and gender != "성별을 선택하세요"
        ):
            customer_info = f"""- 고객 이름: {name}
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
    user_avatar = "https://cdn-icons-png.flaticon.com/512/9131/9131529.png"
    ai_avatar = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ef/ChatGPT-Logo.svg/1024px-ChatGPT-Logo.svg.png"

    for message in st.session_state.message_list:
        role = message["role"]
        content = message["content"]
        avatar = user_avatar if role == "user" else ai_avatar
        display_message(role, content, avatar)

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
                conversation_summary = "상담 중 고객 관심사항과 추가 설명 요청 포함"
                kakao_message = get_kakao_response(
                    st.session_state['script_context'],
                    conversation_summary
                )
                st.session_state['kakao_text'] = "".join(kakao_message)

    with col2:
        if st.button("🆕 새로운 고객 정보 입력하기", use_container_width=True):
            st.session_state.page = "input"
            st.session_state.message_list = []
            st.session_state.script_context = ""
            st.session_state.kakao_text = ""
            st.experimental_rerun()

    # 👉 생성된 카카오톡 문자 출력 (있을 때만 표시)
    if st.session_state.get('kakao_text'):
        st.markdown("### 📩 카카오톡 발송용 문자")
        st.text_area("아래 내용을 복사해 사용하세요.", value=st.session_state['kakao_text'], height=400)
