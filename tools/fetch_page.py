# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/tools/fetch_page.py

import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FetchPageInput(BaseModel):
    url: str = Field()
    page_num: int = Field(0, ge=0)


@tool(args_schema=FetchPageInput)
def fetch_page(url, page_num=0, timeout_sec=10):
    """
    지정된 URL(그리고 page_num)에 해당하는 웹페이지 콘텐츠를 가져오는 도구입니다.

    `status` 와 `page_content`(`title`, `content`, `has_next` 플래그)를 반환합니다.
    status가 200이 아니라면 페이지를 가져오는 과정에서 오류가 발생한 것입니다. (다른 페이지 가져오기를 시도해주세요)

    기본적으로 최대 2,000 토큰 분량의 콘텐츠만 가져옵니다.
    페이지에 더 많은 콘텐츠가 있다면 `has_next` 값이 True가 됩니다.
    계속 읽으려면 동일한 URL에 대해 `page_num` 파라미터를 증가시켜 다시 요청하면 됩니다.
    (페이지 번호는 0부터 시작하므로 다음 페이지는 1입니다)

    한 페이지가 너무 길 경우 **3페이지 이상 조회하지 마세요** (메모리 부담이 커지기 때문)

    Returns
    -------
    Dict[str, Any]:
    - status: str
    - page_content
      - title: str
      - content: str
      - has_next: bool
    """
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.encoding = "utf-8"
    except requests.exceptions.Timeout:
        return {
            "status": 500,
            "page_content": {
                "error_message": "Timeout Error로 인해 페이지를 다운로드할 수 없습니다. 다른 페이지를 가져와 보세요."
            },
        }

    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {
                "error_message": "페이지를 다운로드할 수 없습니다. 다른 페이지를 가져와 보세요."
            },
        }

    try:
        doc = Document(response.text)
        title = doc.title()
        html_content = doc.summary()
        content = html2text.html2text(html_content)
    except:
        return {
            "status": 500,
            "page_content": {
                "error_message": "페이지를 파싱할 수 없습니다. 다른 페이지를 가져와 보세요."
            },
        }

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)
    if page_num >= len(chunks):
        return {
            "status": 500,
            "page_content": {
                "error_message": "page_num 파라미터가 잘못된 것 같습니다. 다른 페이지를 가져와 보세요."
            },
        }
    elif page_num >= 3:
        return {
            "status": 503,
            "page_content": {
                "error_message": "더 많은 page_num 콘텐츠를 읽으면 메모리가 과부하됩니다. 현재까지 확보한 정보만으로 답변을 작성해주세요."
            },
        }
    else:
        return {
            "status": 200,
            "page_content": {
                "title": title,
                "content": chunks[page_num],
                "has_next": page_num < len(chunks) - 1,
            },
        }
