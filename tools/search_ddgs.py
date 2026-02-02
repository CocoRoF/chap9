from itertools import islice
from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field


"""
DDGS Python 라이브러리의 응답 예시
--------------------------------------------
[
    {
        "title": "2025년 한국시리즈 - 위키백과, 우리 모두의 백과사전",
        "snippet": "결과는 LG 트윈스 가 시리즈 전적 4승 1패로 2023년 한국시리즈 이후 2년 만의 통합 우승을 이뤄내 면서 팀 통산 4번째 우승을 기록했다. 한국시리즈 MVP는 시리즈 기간 17타수 9안타 8타점을 기록, 기자단 투표 …",
        "url": "https://ko.wikipedia.org/wiki/2025년_한국시리즈",
    }, ...
]
"""



class SearchDDGSInput(BaseModel):
    query: str = Field(description="검색할 키워드를 입력하세요")


@tool(args_schema=SearchDDGSInput)
def search_ddgs(query, max_result_num=5):
    """
    DDGS 검색을 실행하는 도구입니다.
    검색할 키워드를 입력해 사용해 주세요.
    검색 결과의 각 페이지에 대한 제목, 스니펫(설명), URL이 반환됩니다.
    이 도구에서 얻을 수 있는 정보는 매우 단순화되어 있으며, 경우에 따라 오래된 정보일 수도 있습니다.

    원하는 정보를 찾지 못했다면 반드시 ‘fetch_page’ 도구를 사용해 각 페이지의 실제 내용을 확인해 주세요.
    문맥에 따라 가장 적합한 언어로 검색하세요 (사용자의 언어와 같을 필요는 없습니다).
    예를 들어, 프로그래밍 관련 질문이라면 영어로 검색하는 것이 가장 효과적입니다.

    Returns
    -------
    List[Dict[str, str]]:
    - title
    - snippet
    - url
    """
    res = DDGS().text(query, region="ko-kr", safesearch="off", backend="duckduckgo")
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in islice(res, max_result_num)
    ]
