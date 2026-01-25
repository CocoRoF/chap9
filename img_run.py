"""
LangGraph Agent 그래프 시각화 스크립트

main.py에 정의된 에이전트의 내부 그래프 구조를
이미지 파일로 저장합니다.

개별 도구(search_ddg, fetch_page)를 별도 노드로 표시합니다.
"""

import os
from typing import Annotated, TypedDict

# dotenv 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def create_visualization_graph():
    """
    main.py 에이전트 구조를 시각화하기 위한 커스텀 그래프 생성
    (개별 도구를 별도 노드로 표시)
    """

    # 더미 노드 함수들 (시각화용)
    def summarization_middleware(state):
        """대화 요약 미들웨어"""
        return state

    def model_node(state):
        """LLM 모델 (GPT-5.2 / Claude / Gemini)"""
        return state

    def search_ddg_node(state):
        """DuckDuckGo 검색 도구"""
        return state

    def fetch_page_node(state):
        """웹페이지 내용 가져오기 도구"""
        return state

    def tool_router(state):
        """도구 라우팅 (어떤 도구를 호출할지 결정)"""
        # 시각화용 더미 - 실제로는 model의 tool_calls에 따라 결정
        return "search_ddg"

    def should_continue(state):
        """계속 진행할지 종료할지 결정"""
        return "end"

    # 그래프 구성
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("SummarizationMiddleware", summarization_middleware)
    graph.add_node("model", model_node)
    graph.add_node("search_ddg", search_ddg_node)
    graph.add_node("fetch_page", fetch_page_node)

    # 엣지 추가
    # START -> 미들웨어 -> 모델
    graph.add_edge(START, "SummarizationMiddleware")
    graph.add_edge("SummarizationMiddleware", "model")

    # 모델 -> 도구들 또는 종료 (조건부)
    graph.add_conditional_edges(
        "model",
        should_continue,
        {
            "search_ddg": "search_ddg",
            "fetch_page": "fetch_page",
            "end": END
        }
    )

    # 도구 -> 미들웨어로 돌아감 (루프)
    graph.add_edge("search_ddg", "SummarizationMiddleware")
    graph.add_edge("fetch_page", "SummarizationMiddleware")

    return graph.compile()


def save_graph_image(graph, output_path="agent_graph.png"):
    """
    그래프를 이미지로 저장
    """
    drawable = graph.get_graph()
    png_data = drawable.draw_mermaid_png()

    with open(output_path, "wb") as f:
        f.write(png_data)

    print(f"✅ 그래프 이미지가 저장되었습니다: {output_path}")


def save_graph_as_mermaid(graph, output_path="agent_graph.md"):
    """
    그래프를 Mermaid 마크다운으로 저장
    """
    drawable = graph.get_graph()
    mermaid_code = drawable.draw_mermaid()

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Web Browsing Agent Graph\n\n")
        f.write("## 구성 요소\n\n")
        f.write("| 노드 | 설명 |\n")
        f.write("|------|------|\n")
        f.write("| **SummarizationMiddleware** | 대화 기록 요약 미들웨어 (토큰 한도 초과 시 자동 요약) |\n")
        f.write("| **model** | LLM 모델 (GPT-5.2 / Claude Sonnet 4.5 / Gemini 2.5 Flash) |\n")
        f.write("| **search_ddg** | DuckDuckGo 검색 도구 |\n")
        f.write("| **fetch_page** | 웹페이지 내용 가져오기 도구 |\n\n")
        f.write("## 흐름\n\n")
        f.write("1. 사용자 입력 → SummarizationMiddleware (대화 기록 관리)\n")
        f.write("2. model이 도구 호출 필요 여부 판단\n")
        f.write("3. 도구 호출 시 search_ddg 또는 fetch_page 실행\n")
        f.write("4. 도구 결과를 다시 model로 전달 (루프)\n")
        f.write("5. 충분한 정보 수집 시 최종 응답 생성 후 종료\n\n")
        f.write("## 그래프 다이어그램\n\n")
        f.write("```mermaid\n")
        f.write(mermaid_code)
        f.write("\n```\n")

    print(f"✅ Mermaid 코드가 저장되었습니다: {output_path}")


def print_graph_info(graph):
    """
    그래프 정보 출력
    """
    drawable = graph.get_graph()

    print("\n" + "="*50)
    print("📊 에이전트 그래프 정보")
    print("="*50)

    print("\n🔷 노드 (Nodes):")
    for node in drawable.nodes:
        print(f"   - {node}")

    print("\n🔗 엣지 (Edges):")
    for edge in drawable.edges:
        print(f"   - {edge}")

    print("="*50 + "\n")


if __name__ == "__main__":
    print("🔧 시각화용 커스텀 그래프 생성 중...")
    graph = create_visualization_graph()

    print_graph_info(graph)

    print("📊 그래프 시각화 중...")

    try:
        save_graph_image(graph, "agent_graph.png")
    except Exception as e:
        print(f"⚠️ PNG 저장 실패: {e}")

    try:
        save_graph_as_mermaid(graph, "agent_graph.md")
    except Exception as e:
        print(f"⚠️ Mermaid 저장 실패: {e}")

    print("\n🎉 완료!")
