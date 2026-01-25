import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page


OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GOOGLE_API_KEY = ""


import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn(
        ".env 파일을 통한 API Key 로드에 실패했습니다. main.py 상단에 입력된 API Key를 사용합니다.",
        ImportWarning,
    )

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
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해주세요!"}
        ]
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())


def select_model():
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-5.2":
        return ChatOpenAI(temperature=0, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_web_browsing_agent():
    tools = [search_ddg, fetch_page]
    llm = select_model()

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        max_tokens_before_summary=8000,
        messages_to_keep=10,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        checkpointer=st.session_state["checkpointer"],
        middleware=[summarization_middleware],
        debug=True
    )

    return agent


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_web_browsing_agent()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}
            final_response = ""
            status_container = st.status("🤔 Thinking...", expanded=True)
            response_placeholder = st.empty()

            for stream_mode, data in web_browsing_agent.stream(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config,
                stream_mode=["messages", "updates"]
            ):
                if stream_mode == "updates":
                    for source, update in data.items():
                        if not isinstance(update, dict):
                            continue

                        messages = update.get("messages", [])
                        for msg in messages:
                            if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    with status_container:
                                        st.write(f"🔧 **{tc.get('name', 'tool')}**: `{tc.get('args', {})}`")

                            if source == "tools" and hasattr(msg, 'name'):
                                tool_name = msg.name
                                tool_content = str(msg.content) if hasattr(msg, 'content') else ""

                                with status_container:
                                    st.write(f"✅ **{tool_name}** 완료")
                                    with st.expander(f"📋 {tool_name} 결과 보기", expanded=False):
                                        if len(tool_content) > 2000:
                                            st.code(tool_content[:2000] + "\n... (truncated)", language="text")
                                        else:
                                            st.code(tool_content, language="text")

                elif stream_mode == "messages":
                    chunk, metadata = data

                    if metadata.get("langgraph_node") == "tools":
                        continue

                    if hasattr(chunk, 'content') and chunk.content:
                        if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                            continue
                        final_response += chunk.content
                        response_placeholder.markdown(final_response + "▌")

            status_container.update(label="✅ Complete!", state="complete", expanded=False)

            if final_response:
                response_placeholder.markdown(final_response)
                st.session_state.messages.append({"role": "assistant", "content": final_response})


if __name__ == "__main__":
    main()
