import streamlit as st

# LangChain 1.0.0+ 새로운 Agent API
# create_agent: LangChain 1.0.0+에서 도입된 새로운 에이전트 생성 함수
# 기존의 create_tool_calling_agent + AgentExecutor를 대체함
from langchain.agents import create_agent

# 대화 기록 자동 요약 미들웨어
# ConversationBufferWindowMemory의 대안으로, 토큰 한도에 도달하면 자동으로 대화 내용을 요약
from langchain.agents.middleware import SummarizationMiddleware

# 메모리 관리를 위한 LangGraph Checkpointer
# LangChain 1.0.0+에서는 LangGraph 기반의 checkpointer를 통해 대화 기록을 관리
# 기존의 ConversationBufferWindowMemory를 대체
from langgraph.checkpoint.memory import InMemorySaver
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# 커스텀 도구 임포트
# @tool 데코레이터로 정의된 함수들은 LangChain 1.0.0+에서도 동일하게 사용 가능
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page


###### dotenv(.env)를 사용하지 않는 경우 입력해주세요 ######
OPENAI_API_KEY = ""                                  # 여기에 OpenAI API Key를 입력하세요
ANTHROPIC_API_KEY = ""                               # 여기에 Anthropic API Key를 입력하세요
GOOGLE_API_KEY = ""                                  # 여기에 Google Generative AI API Key를 입력하세요


###### dotenv(.env) 혹은 상단에 정의된 변수를 통해서 API_KEY를 불러옵니다. ######
# 1. dotenv(.env)에서 우선 로드 시도
# 2. 환경변수에 값이 없으면 위에 정의된 변수값을 개별적으로 적용
import os

# dotenv가 설치되어 있으면 .env 파일 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn(
        ".env 파일을 통한 API Key 로드에 실패했습니다. main.py 상단에 입력된 API Key를 사용합니다.",
        ImportWarning,
    )

# .env에 없거나 비어있으면 상단에 정의된 변수값 사용
if not os.getenv("OPENAI_API_KEY") and OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

if not os.getenv("ANTHROPIC_API_KEY") and ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY

if not os.getenv("GOOGLE_API_KEY") and GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

missing_keys = []
if not os.getenv("OPENAI_API_KEY"):
    missing_keys.append("OPENAI_API_KEY")
if not os.getenv("ANTHROPIC_API_KEY"):
    missing_keys.append("ANTHROPIC_API_KEY")
if not os.getenv("GOOGLE_API_KEY"):
    missing_keys.append("GOOGLE_API_KEY")

if missing_keys:
    import warnings
    warnings.warn(
        f"다음 API Key가 설정되지 않았습니다: {', '.join(missing_keys)}. "
        ".env 파일을 사용하거나, main.py 상단에 직접 API Key를 입력해주세요.",
        UserWarning,
    )
################################################


# 시스템 프롬프트 정의
# LangChain 1.0.0+에서는 create_agent의 system_prompt 파라미터로 직접 전달
# 기존의 ChatPromptTemplate 구성 없이 문자열로 간단하게 전달 가능
CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다.
사용 가능한 도구를 활용하여 조사한 정보를 설명해주세요.
이미 알고 있는 정보만으로 답변하지 말고, 가능한 한 검색을 수행한 뒤 답변해주세요.
(사용자가 읽을 페이지를 지정하는 등 특별한 경우는 검색하지 않아도 됩니다.)

검색 결과 페이지만 확인했을 때 정보가 충분하지 않다고 판단되면 다음 옵션을 고려해 시도해 주세요.

- 검색 결과의 링크를 클릭해 각 페이지의 콘텐츠를 열람하고 내용을 확인하세요.
- 한 페이지가 너무 길 경우, 3페이지 이상 스크롤하지 마세요 (메모리 부담 때문).
- 검색 쿼리를 변경한 뒤 다시 검색을 시도하세요.
- 공식 문서뿐 아니라 블로그, 커뮤니티 등 비공식 자료도 함께 참고하세요.

사용자는 매우 바쁘며, 당신만큼 여유롭지 않습니다.
따라서 사용자의 수고를 덜어주기 위해 **직접적인 답변**을 제공해주세요.

=== 나쁜 답변 예시 ===
- 다음 페이지들을 참고하세요.
- 이 페이지들을 보고 코드를 작성할 수 있습니다.
- 다음 페이지가 도움이 될 것입니다.

=== 좋은 답변 예시 ===
- 이 문제의 해결 예시는 다음과 같습니다. -- 여기 코드 제시 --
- 질문에 대한 답은 다음과 같습니다. -- 여기 답변 제시 --

답변 마지막에는 **참조한 페이지의 URL을 반드시 기재**해주세요.
(사용자가 정보를 검증할 수 있도록)

사용자가 사용하는 언어로 답변해주세요.
사용자가 한국어로 질문하면 한국어로, 스페인어로 질문하면 스페인어로 답변해야 합니다.
"""


def init_page():
    st.set_page_config(page_title="Web Browsing Agent", page_icon="🤗")
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")

def init_messages():

    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        # 초기 환영 메시지 설정
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해주세요!"}
        ]
        # LangGraph Checkpointer 초기화
        # InMemorySaver: 메모리 기반 체크포인터 (앱 재시작 시 초기화됨)
        st.session_state["checkpointer"] = InMemorySaver()

        # thread_id: 대화 세션을 구분하는 고유 식별자
        # 동일한 thread_id를 사용하면 이전 대화 내역이 유지됨
        # 새 대화를 시작할 때마다 새로운 thread_id 생성
        st.session_state["thread_id"] = str(uuid.uuid4())


def select_model():
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)

    # 모델 인스턴스 직접 생성 방식 (세부 설정이 필요한 경우)
    # temperature=0으로 설정하여 일관된 응답 생성
    if model == "GPT-5.2":
        return ChatOpenAI(temperature=0, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_web_browsing_agent():
    tools = [search_ddg, fetch_page]
    llm = select_model()

    # 대화 기록 자동 요약 미들웨어 설정
    # ConversationBufferWindowMemory(k=10)의 대안
    # 토큰 한도에 도달하면 오래된 메시지를 자동으로 요약하여 컨텍스트 유지
    summarization_middleware = SummarizationMiddleware(
        model=llm,                    # 요약에 사용할 LLM (에이전트와 동일 모델 사용)
        max_tokens_before_summary=8000,  # 이 토큰 수를 초과하면 요약 시작
        messages_to_keep=10,          # 최근 N개 메시지는 요약하지 않고 유지 (k=10과 유사)
    )

    # LangChain 1.0.0+ create_agent 사용
    agent = create_agent(
        model=llm,                              # LLM 모델 (인스턴스 또는 문자열)
        tools=tools,                            # 사용 가능한 도구 리스트
        system_prompt=CUSTOM_SYSTEM_PROMPT,     # 시스템 프롬프트
        checkpointer=st.session_state["checkpointer"],  # 대화 상태 저장용 체크포인터
        middleware=[summarization_middleware],  # 대화 요약 미들웨어 적용
        debug=True                              # 디버그 모드 (verbose 대체)
    )

    return agent


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_web_browsing_agent()

    # 기존 메시지 표시
    # LangChain 1.0.0+에서는 st.session_state.messages를 직접 관리
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 사용자 입력 처리
    if prompt := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # 에이전트 설정
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

            # 최종 응답 저장
            final_response = ""

            # 중간 단계 표시용 status
            status_container = st.status("🤔 Thinking...", expanded=True)

            # 최종 응답 스트리밍용 placeholder
            response_placeholder = st.empty()

            # stream_mode=["messages", "updates"] 사용
            # - messages: LLM 토큰 스트리밍 (AI 응답 실시간 표시)
            # - updates: 상태 업데이트 (도구 호출 정보, 실행 결과 등)
            for stream_mode, data in web_browsing_agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
                stream_mode=["messages", "updates"]
            ):
                # ========== updates 모드: 도구 호출 및 실행 결과 처리 ==========
                if stream_mode == "updates":
                    for source, update in data.items():
                        # update가 None이거나 dict가 아닌 경우 스킵
                        if not isinstance(update, dict):
                            continue

                        messages = update.get("messages", [])
                        for msg in messages:
                            # 도구 호출 정보 표시 (model 노드에서 AIMessage에 tool_calls가 있는 경우)
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    with status_container:
                                        st.write(f"🔧 **{tc.get('name', 'tool')}**: `{tc.get('args', {})}`")

                            # 도구 실행 결과 표시 (tools 노드에서 ToolMessage)
                            if source == "tools" and hasattr(msg, 'name'):
                                tool_name = msg.name
                                tool_content = str(msg.content) if hasattr(msg, 'content') else ""

                                with status_container:
                                    st.write(f"✅ **{tool_name}** 완료")
                                    # 결과 내용을 expander로 표시
                                    with st.expander(f"📋 {tool_name} 결과 보기", expanded=False):
                                        # 결과가 너무 길면 잘라서 표시
                                        if len(tool_content) > 2000:
                                            st.code(tool_content[:2000] + "\n... (truncated)", language="text")
                                        else:
                                            st.code(tool_content, language="text")

                # ========== messages 모드: AI 응답 토큰 스트리밍 ==========
                elif stream_mode == "messages":
                    chunk, metadata = data

                    # 도구 노드에서 오는 메시지는 updates에서 처리했으므로 스킵
                    if metadata.get("langgraph_node") == "tools":
                        continue

                    # AI 응답 토큰 스트리밍 (tool_call_chunks가 없는 경우만)
                    if hasattr(chunk, 'content') and chunk.content:
                        if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                            continue
                        final_response += chunk.content
                        response_placeholder.markdown(final_response + "▌")

            # status 완료 처리
            status_container.update(label="✅ Complete!", state="complete", expanded=False)

            # 최종 응답 표시 (커서 제거)
            if final_response:
                response_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})


if __name__ == "__main__":
    main()
