from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from functools import lru_cache
import streamlit as st
import os
from dotenv import load_dotenv

# ======================== 설정 ========================
load_dotenv(dotenv_path=".envfile", override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# ======================== 전역 저장소 ========================
store = {}

# ======================== 전역 프롬프트 ========================
SYSTEM_PROMPT_SCRIPT = (
    """
    [상담 스크립트 출력 형식 - 반드시 아래 순서와 마크다운 제목 형식을 지켜주세요]
    각 단계는 다음과 같이 출력하세요:
    - `#### 1. 첫 인사 및 친근한 접근`
    - `#### 2. 상담 목적 확인 및 공감 형성`
    - `#### ✅ 2-1. 상담 목적 확인 – 고객 응대 버전`
      - 아래 두 상황에 대한 멘트를 반드시 모두 출력하세요:
        - `💬 [상황 A: 기억을 못함]` → 고객이 상담 신청이나 이유를 기억하지 못할 때 사용하는 멘트
        - `💬 [상황 B: 바쁨 또는 거절]` → 고객이 지금 바쁘거나 '괜찮다'며 거절할 때 사용하는 멘트. 보장 내용을 카카오톡 메세지로 발송해준다는 내용으로 마무리.
    - `#### 3. 기존 보험 체크 및 간단 분석`
    - `#### 4. 새로운 보험 상품 설명`
    - `#### 5. 맞춤 제안 및 고객 중심 상담`
    - `#### 6. 가입 유도 및 부드러운 마무리`

    [작성 스타일]
    - 각 항목 제목은 반드시 마크다운 형식(`####`)으로 작성하고, 하단에 자연스럽게 말하듯 이어주세요.
    - 고객 이름은 중간중간 자연스럽게 포함시켜 친근함을 표현해주세요.
    - 문장은 실제 상담원이 전화로 말하듯 구어체로 작성해주세요. 번역투/딱딱한 표현은 피해주세요.
    - 고객의 상황과 감정에 공감하는 표현을 포함해주세요. 현실적인 말투와 예시, 감정 표현이 중요합니다.
    - 이모지(😊 🙇‍♀️ 👍 😅 등)는 너무 과하지 않게, 자연스럽게 사용해주세요.
    - `2-1` 항목에서는 `💬 [상황 A: ...]`와 `💬 [상황 B: ...]`를 반드시 **모두 출력**해주세요.
    - 전체 흐름이 자연스럽게 이어지도록 각 단계 사이를 부드럽게 연결해주세요.
    - 마지막 단계는 선택이 부담스럽지 않도록 부드럽게 마무리하는 표현으로 작성해주세요. 그리고 상담 내용을 카카오톡 메세지로 발송해준다는 내용으로 스크랩트를 마무리해주세요.
    
    [연령대 및 성별 반영 지침]
    - 고객의 연령대와 성별에 따라 상담 톤과 설명 포인트를 조절하세요.
    - 예시
    - 20~30대 고객: 부담을 줄이는 방향, 가벼운 보장, 실속형 상담을 강조하세요.
    - 40~50대 고객: 가족 보호, 건강 관리의 중요성을 자연스럽게 언급하세요.
    - 60대 이상 고객: 노후 대비, 간병, 치매 보장의 필요성을 공감하며 설명하세요.
    - 남성 고객: 실용적이고 간결한 설명을 선호하므로, 핵심 보장을 중심으로 전달하세요.
    - 여성 고객: 간병, 생활 속 위험 대비 등을 강조하며, 공감 표현을 조금 더 활용하세요.
    - 단, 성별에 대한 고정관념은 피하고, 상담이 자연스럽고 친근하게 이어지도록 하세요.

    [출력 목표]
    상담 상황 입력을 바탕으로, 위 1~6단계 형식을 그대로 따르며,
    현실적인 전화 상담 멘트로 구성된 스크립트를 생성하세요.
    `2-1` 단계에서는 두 가지 상황(A, B)에 대해 각각 따로 쓸 수 있도록 **항상 두 버전을 모두 출력**하세요.
    전체 멘트는 상담원이 바로 사용할 수 있을 만큼 자연스럽고 실용적이어야 합니다.
    
    **스크립트 출력 이후 상황에 맞는 핵심 팁**을 제시하세요.

    [출력 형식 지침]
    - 반드시 아래 형식을 따르세요:
    📌 상담 TIP
    ▶️ (간결하고 명확한 팁)
    ▶️ (간결하고 명확한 팁)
    ▶️ (간결하고 명확한 팁)

    - 각 팁은 1~2문장으로 구체적으로 작성하세요.
    - 너무 일반적인 조언이 아니라, **해당 상담 상황**에 맞는 팁을 제공하세요.
    - 상담 멘트 활용법, 고객 대응 전략, 설득 스킬, 주의사항 등을 포함할 수 있습니다.
    """
)

SYSTEM_PROMPT_CHATBOT = (
    """
    당신은 보험 전화 상담원을 지원하는 AI 어시스턴트입니다.
    이미 생성된 상담 스크립트를 바탕으로, 상담원이 추가 질문이나 멘트 보완 요청을 하면
    현현실적이고 실무에 도움이 되는 답변을 제공해야 합니다.

    [역할]
    - 상담원이 고객과의 대화를 더 잘 이끌어갈 수 있도록 보조합니다.
    - 추가 멘트 제안, 고객 반응에 따른 대응 방법, 상담 흐름 조언 등을 제공합니다.
    - 상담원이 요청할 경우, 지금까지 대화를 반영하여 새로운 상담 스크립트를 작성합니다.
    - 보험 상품, 상담 기법, 대화 스킬 등 실무적인 팁을 알려줍니다.

    [답변 지침]
    1. 질문에 대해 구체적이고 상담원의 입장에 공감하며 답변해주세요.
    2. 필요할 경우, 실제 상담에 활용할 수 있는 자연스러운 멘트를 예시로 제시하세요.
    3. 상담원이 요청하면, 지금까지 대화 내용과 고객 정보를 반영하여 새로운 스크립트를 작성하세요.
    4. 문장은 구어체로 작성하며, 친절하고 신뢰감 있는 톤을 유지하세요.
    5. 고객의 입장을 공감하는 표현을 적절히 포함하세요.
    6. 모호하거나 애매한 질문일 경우, 상담에서 활용할 수 있는 보완 멘트를 제안하세요.
    7. 이모지는 자연스럽게 사용하되 과도하게 넣지 마세요.
    8. 단순한 멘트 제시에 그치지 말고, 왜 그런 방식이 효과적인지 간단한 설명을 함께 덧붙여 상담사에게 실질적인 학습이 되도록 도와주세요.
    9. 상담 흐름을 자연스럽게 이어가기 위한 전략(예: 질문 유도, 감정 리프레이밍 등)을 제시해 주세요.
    
    [형식 지침]
    - 상담 멘트를 제시할 때는 다음 형식을 지켜주세요:
    **👉 상담 멘트 예시**
    > \"여기에 실제 상담에서 활용할 수 있는 자연스러운 멘트를 작성하세요.\"
    - 멘트 앞에는 반드시 '**👉 상담 멘트 예시**' 라는 제목을 넣어 시각적으로 구분하세요.
    - 멘트는 인용구 형태(`> `)로 작성해 주세요.
    - 멘트 이후에는 간단한 활용 팁이나 주의사항을 덧붙이세요.

    질문을 입력받으면 위 지침에 따라 현실적이고 실용적인 답변을 제공하세요.
    """
)

# ======================== 모델 호출 ========================
@lru_cache(maxsize=1)
def get_llm(model='gpt-4.1-mini'):
    return ChatOpenAI(model=model)

# ======================== 세션 관리 ========================
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# ======================== 랜덤 고객정보 생성 ========================
def get_random_customer_info():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 보험 영업을 위한 가상의 고객 정보를 생성하는 AI 어시스턴트입니다.
        
        [출력 지침]
        보험 상담 고객 정보를 다음 형식으로 랜덤 생성하세요:
        - 고객 이름: 자연스러운 한글 이름을 생성하세요.
        - 연령대: 20대, 30대, 40대, 50대, 60대, 70대 이상 중 하나
        - 성별: 남성 또는 여성
        - 기존 보험 상태: 기존에 가입하거나 보유한 보험을 구체적으로 작성해주세요.
        - 관심 보험: 가상의 고객이 관심을 갖고 있는 보험을 구체적으로 작성해주세요.
        - 고객 반응: 상황과 관련된 현재 고객의 주요 생각을 구체적으로 작성해주세요.
        - 기타 상황: 현재 고객과 관련된 기타 상황을 구체적으로 작성해주세요.

        출력은 반드시 다음처럼 해주세요:

        고객 이름: (이름)
        연령대: (연령대)
        성별: (성별)
        기존 보험 상태: (텍스트)
        관심 보험: (텍스트)
        고객 반응: (텍스트)
        기타 상황: (텍스트)
        """),
        ("human", "랜덤 고객 정보를 생성해 주세요.")
    ])

    chain = prompt_template | get_llm() | StrOutputParser()

    result = chain.invoke({})

    # 결과 파싱
    lines = result.splitlines()
    info = {}
    for line in lines:
        if line.startswith("고객 이름:"):
            info['name'] = line.replace("고객 이름:", "").strip()
        elif line.startswith("연령대:"):
            info['age_group'] = line.replace("연령대:", "").strip()
        elif line.startswith("성별:"):
            info['gender'] = line.replace("성별:", "").strip()
        elif line.startswith("기존 보험 상태:"):
            info['insurance_status'] = line.replace("기존 보험 상태:", "").strip()
        elif line.startswith("관심 보험:"):
            info['interest'] = line.replace("관심 보험:", "").strip()
        elif line.startswith("고객 반응:"):
            info['reaction'] = line.replace("고객 반응:", "").strip()
        elif line.startswith("기타 상황:"):
            info['etc'] = line.replace("기타 상황:", "").strip()

    return info

# ======================== 스크립트 생성 ========================
def get_script_response(name, age_group, gender, insurance_status, interest, reaction, etc):
    try:
        # 1️⃣ 고객 정보 포맷 구성
        customer_info = (
            f"- 고객 이름: {name}\n"
            f"- 연령대: {age_group}\n"
            f"- 성별: {gender}\n"
            f"- 기존 보험: {insurance_status}\n"
            f"- 관심 보험: {interest}\n"
            f"- 고객 반응: {reaction}\n"
            f"- 기타 상황: {etc}"
        )
        
        # 상담원 이름 불러오기 (로그인 시 저장된 값)
        consultant_name = st.session_state.get('user_name', '상담원')

        # dynamic_prompt 생성
        dynamic_prompt = SYSTEM_PROMPT_SCRIPT.replace("{상담원 이름}", consultant_name)
        
        dynamic_prompt = f"""
        당신은 보험 민원 대응을 전문으로 하는 AI 상담 지원 도우미입니다.
        상담원이 입력한 민원 상황과 고객 감정 상태를 바탕으로, 고객의 불만을 효과적으로 완화하고 신뢰를 줄 수 있는 **맞춤형 응대 스크립트**와 실무에 도움이 되는 **상담 TIP**을 함께 제공하세요.
        실제 사람이 말하듯 자연스럽고 실용적인 멘트를 작성해야 하며, 상담원이 현장에서 그대로 사용할 수 있을 정도로 현실적이어야 합니다.
        고객 이름과 상담원 이름을 혼동하지 말고, 반드시 각 정보에 맞게 사용하세요.
        
        ⚠️ 절대 지침
        - 상담원 이름은 반드시 아래 [상담원 정보]의 이름만 사용하세요.
        - 상담원 이름을 임의로 생성하거나 변경하지 마세요.
        - 고객 이름은 반드시 [고객 정보]의 이름만 사용하세요.
        - 다른 이름, 가상의 이름을 절대 생성하지 마세요.

        [상담원 정보]
        - 상담원 이름: {consultant_name}

        [고객 정보]
        {customer_info}

        - 스크립트의 시작 부분에서는 상담원이 본인의 이름을 말하며 밝게 인사하도록 작성하세요.
        - 예시: "안녕하세요, 저는 굿리치 상담사 **{consultant_name}**입니다!"
        
        {SYSTEM_PROMPT_SCRIPT}
        """
        
        # 체인 구성
        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{customer_info}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="customer_info",
            history_messages_key="chat_history",
        )

        result = chain.invoke(
            {"customer_info": customer_info},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        return iter([result])
    
    except Exception as e:
        st.error("🔥 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 대화 챗봇 ========================
def get_chatbot_chain():
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_CHATBOT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    return prompt | get_llm() | StrOutputParser()

def get_chatbot_response(user_message, script_context=""):
    try:
        full_input = f"[현재 상담 스크립트 요약]\n{script_context}\n\n[질문]\n{user_message}"
        chain = RunnableWithMessageHistory(
            get_chatbot_chain(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )
        result = chain.invoke(
            {"input": full_input},
            config={"configurable": {"session_id": st.session_state.session_id}}
        )
        return iter([result])
    except Exception as e:
        st.error("🔥 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])

# ======================== 카카오톡 문자 발송 ========================
def generate_conversation_summary(message_list):
    summary_points = []
    for message in message_list:
        if message['role'] == 'user':
            summary_points.append(f"- 상담원 요청: {message['content']}")
        elif message['role'] == 'ai' and "👉 상담 멘트 예시" in message['content']:
            lines = message['content'].split('\n')
            for line in lines:
                if line.startswith("> "):
                    summary_points.append(f"- 제안 멘트: {line[2:]}")
    return "\n".join(summary_points)
    
def get_kakao_response(script_context, message_list):
    try:
        conversation_summary = generate_conversation_summary(message_list)

        dynamic_prompt = f"""
            [상담 요약]
            {script_context}

            [추가 대화 요약]
            {conversation_summary}

            ⚠️ 반드시 위 상담 요약과 추가 대화 요약 내용을 반영하여 고객 발송용 카카오톡 메시지를 작성하세요.
            
            - 당신은 보험 상담 후 고객에게 발송할 카카오톡 메시지를 작성하는 상담사입니다.
            - 상담 내용을 바탕으로 고객 성향에 맞게 다음 [출력 형식]과 [작성 지침]에 따라 총 **3가지 유형**의 메시지를 작성하세요.

            "[출력 형식]\n"
            "각 메시지는 아래 제목과 형식을 반드시 지켜 작성하세요.\n\n"

            "### 1️⃣ 전문성 강조형\n"
            "(보험 전문가로서 핵심 보장 내용과 필요한 이유를 논리적으로 전달하는 메시지)\n\n"

            "### 2️⃣ 감성형\n"
            "(따뜻하고 배려 있는 말투로, 고객의 마음을 편안하게 해주는 메시지)\n\n"

            "### 3️⃣ 실제 사례형\n"
            "(실제 보험금 지급 사례나 주변 사례를 언급하며 필요성을 자연스럽게 강조하는 메시지)\n\n"
            
            [작성 지침]            
            1. 각 메시지는 **15줄 내외**로 작성하세요.
            2. 문장은 반드시 **문장 단위로 줄바꿈**하여 가독성을 높이세요.
            2-1. 한 문장이 너무 길어져도 **적절하게 줄바꿈**하여 가독성을 높이세요.
            2-2. 내용이 바뀌는 문단은 반드시 **두 번 줄바꿈**하여 가독성을 높이세요.
            3. 고객 이름을 자연스럽게 포함하고, 상황에 맞는 맞춤형 표현을 사용하세요.
            4. 문장은 정중하면서도 부담 없는 톤으로 작성하세요.
            5. 상담한 보험의 구체적인 내용(예: 치매보험의 주요 보장, 간병보험의 활용 사례 등)을 간단히 언급하세요.
            6. 고객이 이해하기 쉽게, 너무 추상적인 표현은 피하고 **실질적인 도움이 되는 설명**을 포함하세요.
            7. 상담한 보험 종류, 보완이 필요한 내용, 고객이 관심을 보인 내용용 등을 반영하세요.
            8. 가입을 강요하지 말고, '편하게 문의 주세요'와 같은 표현으로 마무리하세요.
            9. 이모지는 과하지 않게 사용해주세요.
        """

        chain = RunnableWithMessageHistory(
            ChatPromptTemplate.from_messages([
                ("system", dynamic_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ]) | get_llm() | StrOutputParser(),
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

        kakao_session_id = f"{st.session_state.session_id}_kakao"

        result = chain.invoke(
            {"input": "카카오톡 메시지를 생성해 주세요."},
            config={"configurable": {"session_id": kakao_session_id}}
        )
        return iter([result])

    except Exception as e:
        st.error("🔥 카카오톡 메시지 생성 중 오류가 발생했습니다. 콘솔 로그를 확인해 주세요.")
        print("🔥 예외:", e)
        return iter(["❌ 오류가 발생했습니다. 관리자에게 문의해 주세요."])
