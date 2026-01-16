from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import os
import re
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from elasticsearch import Elasticsearch
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 상태 정의
class Graph_State(TypedDict):
    question: str
    skip_retrieve: bool
    context: List[Document]
    answer: str
    hallucination: str
    retry_cnt: int
    es_client: Optional[Elasticsearch]
    embedding_model: Optional[HuggingFaceEmbeddings]
    llm: Optional[ChatGroq]
    one_way: bool

# 실행 준비
def init_node(state: Graph_State) -> dict:
    current_state = dict(state)

    default_state = {
        "question": "",
        "skip_retrieve": False,
        "context": [],
        "answer": "",
        "hallucination": "",
        "retry_cnt": 0,
        "es_client": None,
        "embedding_model": None,
        "llm": None,
        "one_way": False
    }

    merge_state = default_state
    merge_state.update(current_state)

    # 임베딩 모델 로드
    try:
        import torch
        load_dotenv()
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                'device': device
            },
            encode_kwargs={
                'normalize_embeddings': True,    # 벡터 정규화
            }
        )
        
        merge_state["embedding_model"] = embedding_model
        logger.debug(f"임베딩 모델({EMBEDDING_MODEL_NAME}) 로드 성공")
    except Exception as e:
        logger.error(f"임베딩 모델({EMBEDDING_MODEL_NAME}) 로드 실패: {e}")
        exit(1)

    # Elasticsearch 연결
    try:
        es_host = os.getenv("ES_HOST", "http://localhost:9200")
        es = Elasticsearch(es_host)
        merge_state["es_client"] = es
        logger.debug(f"Elasticsearch 연결 성공: {es_host}")
    except Exception as e:
        logger.error(f"ES 연결 실패: {e}"); exit(1)

    # LLM 모델 로드
    try:
        llm = ChatGroq(
            model_name=os.getenv("GROQ_API_MODEL"),
            temperature=float(os.getenv("GROQ_API_TEMPERATURE", "0.7")),
            api_key=os.getenv("GROQ_API_KEY")
        )
        merge_state["llm"] = llm
        logger.debug(f"LLM 모델 로드 성공")
    except Exception as e:
        logger.error(f"LLM 모델 로드 실패: {e}")
        exit(1)

    return merge_state

# 사용자에게 질문 받기
def input_node(state: Graph_State) -> dict:
    if state["one_way"] == False:
        user_input = input("증상을 입력하세요(종료하고 싶다면 quit): ")
        if user_input.lower() == "quit":
            return {"question": "quit"}
    else:
        user_input = state["question"]

    llm = state["llm"]
    messages = [
        SystemMessage(content=
                        """
                        당신은 질문이 반려견과 관련 있는 질문인지 판단하는 동물병원의 직원입니다.
                        다음은 사용자가 당신에게 물어보는 질문입니다.
                        해당 질문이 반려견이 가진 질병의 증상 혹은 반려견의 성장에 관한 질문이라면 'yes'로 답변해 주세요.
                        만약 질문이 반려견과 관련이 없다면 'no'라고 답변해 주세요.
                        """
        ),
        HumanMessage(content=user_input)
    ]

    response = llm.invoke(messages)  # LLM에게 메시지 전달
    judgment = response.content.strip()

    # 출력 로그
    logger.debug(f"[LLM 응답] {judgment}")

    if judgment.lower() == "no":
        return {"question": user_input, "skip_retrieve": True}

    return {"question": user_input, "skip_retrieve": False, "retry_cnt": 0}


def retriever_node(state: Graph_State) -> dict:
    es = state["es_client"]
    query = state["question"]
    model = state["embedding_model"]
    index_name = os.getenv("ES_INDEX", "my_index")

    logger.info(f"\n문서 검색 중...")
    
    # 1. 키워드 검색으로 상위 200개 후보군 ID 추출
    keyword_query = {
        "size": 200,
        "_source": False, # ID만 필요하므로 본문은 제외하여 속도 향상
        "query": {
            "match": {
                "text": {
                    "query": query
                }
            }
        }
    }
    
    try:
        kw_response = es.search(index=index_name, body=keyword_query)
        candidate_hits = kw_response["hits"]["hits"]
        
        if not candidate_hits:
            logger.warning("키워드 검색 결과가 없습니다.")
            return {"context": []}

        # 후보 ID 리스트 생성
        candidate_ids = [hit["_id"] for hit in candidate_hits]
        
        # 2. 질문 임베딩 생성
        query_vector = model.embed_query(query)

        # 3. 추출된 ID 내에서만 벡터 검색 (Filter 사용)
        vector_search_query = {
            "size": 5, # 최종적으로 필요한 5개만 추출
            "knn": {
                "field": "vector",
                "query_vector": query_vector,
                "k": 15,            # 내부적으로 고려할 이웃 수
                "num_candidates": 200, 
                "filter": {         # 핵심: 1단계에서 찾은 ID들로만 범위를 제한
                    "ids": {
                        "values": candidate_ids
                    }
                }
            }
        }

        v_response = es.search(index=index_name, body=vector_search_query)
        final_hits = v_response["hits"]["hits"]

        retrieved_docs = []
        for hit in final_hits:
            source = hit["_source"]
            doc = Document(
                page_content=source.get("text", ""),
                metadata={
                    "id": hit["_id"],
                    "source": source.get("source", "N/A"),
                    "score": hit["_score"] # Elasticsearch가 계산한 최종 유사도 점수
                }
            )
            retrieved_docs.append(doc)
            
        logger.debug(retrieved_docs)
        return {"context": retrieved_docs}

    except Exception as e:
        logger.error(f"검색 노드 실행 중 오류 발생: {e}")
        return {"context": []}

# context와 question을 가지고 답변 생성
def generate_node(state: Graph_State) -> dict:
    llm = state["llm"]
    question = state["question"]
    logger.info("\n답변 생성 중...")

    if state.get("skip_retrieve", False):    # 검색 건너뜀
        messages = [
            SystemMessage(content="다음 사용자의 질문에 간단히 한국어로 답변해주세요."),
            HumanMessage(content=question)
        ]
        response = llm.invoke(messages)
        answer = response.content.strip()
        logger.debug(f"[응답] {answer}")
        return {"answer": answer}

    context_docs = state.get("context", [])
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    base_guideline=f"""
                        1. 사용자의 질문을 읽고 이해합니다.
                        2. 제공된 문서에서 질문을 대답하는 데 도움이 되는 정보를 찾습니다.
                        3. 사용자의 질문에 맞게 정보를 한국어로 친절하게 답변합니다.
                        4. 문서에 없는 정보는 **절대** 추가하지 말고, 모른다고 솔직하게 말합니다.
                    """
    
    if state.get("retry_cnt", 0) > 0:
        hallucination = state["hallucination"]
        addition_guideline = f"""
                    {base_guideline}
                    5. 이전에 생성된 답변에서 다음과 같은 할루시네이션이 발견되었습니다:
                    {hallucination}
                    6. 위의 할루시네이션을 다시 생성하지 않도록 유의하며 답변을 생성해 주세요.
                    """
        guideline = addition_guideline
    else:
        guideline = base_guideline
        
    messages = [
        SystemMessage(content=f"""
                        당신은 반려견 전문 수의사입니다. 아래 지침 단계에 맞게 행동해 주세요.
                        [지침]
                        {guideline}
                        **중요! ###이나 표, **같은 마크다운 형식으로 대답하지 마세요!**
                        """),
        HumanMessage(content=f"[질문] {question}\n\n[참고 문서]\n{context_text}")
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()

    return {"answer": answer}

def check_hallucination_node(state: Graph_State) -> dict:
    if state["skip_retrieve"] == True:
        return {"hallucination": "none"}

    if state["retry_cnt"] >= 1:
        return {"hallucination": "fail"}

    llm = state["llm"]
    context_docs = state.get("context", [])
    context_text = "\n\n".join([doc.page_content for doc in context_docs])
    answer = state["answer"]

    messages = [
        SystemMessage(content="""
                        당신은 반려견 전문 수의사의 답변을 검토하는 직원입니다. 아래 지침 단계에 맞게 행동해 주세요.
                        [지침]
                        1. 참고 문서를 읽습니다.
                        2. 수의사에 의해 생성된 답변을 읽습니다.
                        3. 답변의 내용에 참고 문서에는 없는 정보 즉, **할루시네이션**이 포함되어 있는지 검토합니다.
                            - 정보가 사소히 다르거나 표현이 부드럽게 바뀌었더라도, 해당 내용이 문서 내 의미와 관련성이 있으면 환각이 아닙니다.
                            - 지엽적인 표현, 파생어, 동의어 또는 설명 방식의 차이만으로 환각이라고 판단하지 말아주세요. 실제로 문맥에 없는 핵심 정보만 환각으로 평가합니다.
                            - 문헌에 근거하지 않은 주장이나, 원문에 없는 정보를 추가로 만든 경우에만 환각 처리하세요.
                        4. 할루시네이션이 없다면 'none', 있다면 정확히 어떤 부분에 할루시네이션이 있었는지 답변합니다.
                        """),
        HumanMessage(content=f"[참고 문서]\n{context_text}\n\n[생성된 답변]\n{answer}")
    ]

    logger.info("\n할루시네이션 검증 중...")
    response = llm.invoke(messages)
    hallucination = response.content.strip()

    return {"hallucination": hallucination}

def remove_markdown(text: str) -> str:
    """마크다운 형식을 제거하는 함수"""
    import re

    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # 코드 블록 제거 (```...```)
    text = re.sub(r'`([^`]+)`', r'\1', text)  # 인라인 코드 제거 (`...`)
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 굵은 글씨 제거 (**...**)
    text = re.sub(r'__([^_]+)__', r'\1', text)  # 굵은 글씨 제거 (__...__)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 기울임 글씨 제거 (*...*)
    text = re.sub(r'_([^_]+)_', r'\1', text)  # 기울임 글씨 제거 (_..._)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # 제목 제거 (# ## ###)
    text = re.sub(r'^[-*]{3,}$', '', text, flags=re.MULTILINE)  # 수평선 제거 (--- ***)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)  # 앞에 공백이 있는 헤더도 제거

    return text

# 최종 답변 출력 후 재시작
def output_node(state: Graph_State) -> dict:
    answer = state.get("answer")
    final_answer=remove_markdown(answer)

    print(f"[응답] {final_answer}")
    return {"answer": final_answer}

def fallback_node(state: Graph_State) -> dict:
    answer = "죄송합니다. 현재 질문에 관한 정보를 찾지 못 했습니다. 더 구체적인 질문으로 다시 시도하거나 다른 질문을 해 주세요."
    print(f"[응답] {answer}")
    return {"answer": answer}

# 워크플로우 정의 (MemorySaver 사용하기)
def workflow():
    graph = StateGraph(Graph_State)

    graph.add_node("init_node", init_node)
    graph.add_node("input_node", input_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("check_hallucination_node", check_hallucination_node)
    graph.add_node("output_node", output_node)
    graph.add_node("fallback_node", fallback_node)

    graph.set_entry_point("init_node")

    graph.add_edge("init_node", "input_node")

    def dedicate_quit(state: Graph_State) -> str:
        if state["question"].lower() == "quit":
            print("프로그램을 종료합니다.")
            return "QUIT"
        else:
            if state["skip_retrieve"] == False:
                return "RELEVANT"
            else:
                print("검색 단계를 건너뜁니다.")
                return "UNRELEVANT"

    graph.add_conditional_edges("input_node",
                                dedicate_quit,
                                    {
                                        "RELEVANT": "retriever_node",
                                        "UNRELEVANT": "generate_node",
                                        "QUIT" : END
                                    }
    )
    graph.add_edge("retriever_node", "generate_node")
    graph.add_edge("generate_node", "check_hallucination_node")

    def dedicate_retry(state: Graph_State) -> str:
        if state["hallucination"] == "none":
            logger.debug("할루시네이션 없음")
            return "NONE"
        else:
            if state["retry_cnt"] >= 1:
                logger.debug("재시도 횟수 초과")
                return "FAIL"
            state["retry_cnt"] += 1
            logger.debug(f"할루시네이션 발견: {state['hallucination']}. 답변을 재생성합니다.")
            return "EXIST"
            

    graph.add_conditional_edges("check_hallucination_node",
                                dedicate_retry,
                                    {
                                        "NONE": "output_node",
                                        "EXIST": "generate_node",
                                        "FAIL": "fallback_node"
                                    }
    )
    
    def dedicate_again(state: Graph_State) -> str:
        if state["one_way"] == True:
            return "EXTERNAL"
        else:
            return "INTERNAL"
    graph.add_conditional_edges("output_node",
                                dedicate_again,
                                    {
                                        "EXTERNAL": END,
                                        "INTERNAL": "input_node"
                                    }
    )
    graph.add_edge("fallback_node", "input_node")

    return graph.compile()

# PNG 이미지 생성
def draw_workflow(app):
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("my_agent_workflow.png", "wb") as f:
            f.write(png_data)

        print("그래프가 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"이미지 생성 중 오류가 발생했습니다: {e}")

# 외부 호출용 함수
async def run(querry):
    app = workflow()
    initial_state = {"question" : querry, "one_way": True}
    final_state = app.invoke(initial_state)
    return final_state["answer"]

def main():
    app = workflow()
    # draw_workflow(app)

    initial_state = {}
    app.invoke(initial_state)

if __name__ == "__main__":
    main()