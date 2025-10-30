from langchain_milvus import Milvus
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLLM
from elasticsearch import Elasticsearch

# 상태 정의
class Graph_State(TypedDict):
    question: str
    skip_retrieve: bool
    context: List[Document]
    answer: str
    quality: str
    retry_cnt: int
    milvus: Optional[Milvus]
    embedding_model: Optional[Embeddings]
    llm: Optional[BaseLLM]

# 실행 준비
def init_node(state: Graph_State) -> dict:
    # 데이터 초기화
    initial_state = {
        "question": "",
        "skip_retrieve": False,
        "context": [],
        "answer": "",
        "quality": "",
        "retry_cnt": 0,
        "milvus": None,
        "embedding_model": None,
        "llm": None,
    }

    # 임베딩 모델 로드
    try:
        load_dotenv()
        EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

        embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={
                'device': 'cuda' 
            },
            encode_kwargs={
              'normalize_embeddings': True,    # 벡터 정규화
            }
        )
        
        initial_state["embedding_model"] = embedding_model
        print(f"임베딩 모델({EMBEDDING_MODEL_NAME}) 로드 성공")
    except Exception as e:
        print(f"임베딩 모델({EMBEDDING_MODEL_NAME}) 로드 실패: {e}")
        exit(1)

    # MilvusDB 연결
    try:
        MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "my_collection")
        MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
        MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

        milvus = Milvus(
            embedding_function=embedding_model,
            collection_name=MILVUS_COLLECTION_NAME,
            connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
            vector_field="vector", # 벡터 필드 이름
            index_params={"metric_type": "COSINE"}
        )
        initial_state["milvus"] = milvus
        print(f"Milvus 연결 성공: {MILVUS_HOST}")
    except Exception as e:
        print(f"Milvus 연결 실패: {e}")
        exit(1)

    # LLM 모델 로드
    try:
        GROQ_API_KEY=os.getenv("GROQ_API_KEY")
        LLM_MODEL=os.getenv("GROQ_API_MODEL","openai/gpt-oss-20b")
        TEMP=os.getenv("GROQ_API_TEMPERATURE", "0.7")

        # 모델 설정
        llm = ChatGroq(model_name=LLM_MODEL, temperature=TEMP, api_key=GROQ_API_KEY)
        initial_state["llm"] = llm
        print(f"LLM 모델({LLM_MODEL}) 로드 성공")
    except Exception as e:
        print(f"LLM 모델({LLM_MODEL}) 로드 실패: {e}")
        exit(1)

    return initial_state

# 사용자에게 질문 받기
def input_node(state: Graph_State) -> dict:
    user_input = input("증상을 입력하세요(종료하고 싶다면 quit): ")
    if user_input.lower() == "quit":
        return {"question": "quit"}
    
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

    print("\n[LLM에게 요약 요청 중...]")
    response = llm.invoke(messages)  # LLM에게 메시지 전달
    judgment = response.content.strip()

    # 출력 로그
    print(f"[LLM 응답] {judgment}")

    if judgment.lower() == "no":
        print("질문이 반려견과 관련없다고 판단")
        return {"question": user_input, "skip_retrieve": True}

    return {"question": user_input, "skip_retrieve": False, "retry_cnt": 0}

# 벡터DB에서 유사도 검색
def elastic_retrieve(query, top_k=50):
    es = Elasticsearch("http://localhost:9200")
    elastic_query = {
        "query": {
            "match": {
                "text": {
                    "query": query,
                    "analyzer": "korean_analyzer"
                }
            }
        },
        "size": top_k
    }

    results = es.search(index="my_index", body=elastic_query)
    hits = results["hits"]["hits"]

    print(f"\n[ElasticSearch 검색 결과] 총 {len(hits)}건 발견 (query='{query}')")
    for idx, hit in enumerate(hits, start=1):
        doc_id = hit["_id"]
        score = hit.get("_score", 0.0)
        text_preview = hit["_source"].get("text", "")[:100].replace("\n", " ")
        print(f"  {idx}. ID={doc_id}, SCORE={score:.4f}, TEXT={text_preview}")

    docs = [
        (
            hit["_id"],
            hit["_source"].get("text", ""),
            hit.get("_score", 0.0)
        )
        for hit in hits
    ]
    return docs

def retriever_node(state: Graph_State) -> dict:
    milvus_vectorstore = state["milvus"]
    query = state["question"]

    print(f"\n[질의] {query}")
    print("ElasticSearch로 1차 후보군 검색 중...")

    # (1) ElasticSearch에서 후보군 가져오기
    candidate_docs = elastic_retrieve(query, top_k=100)

    if not candidate_docs:
        print("[경고] ElasticSearch에서 후보군을 찾지 못했습니다. → Milvus만으로 검색 수행")
        results = milvus_vectorstore.similarity_search_with_score(query, k=15)
        combined_scores = [(doc, score * 100) for doc, score in results]
    else:
        candidate_ids = [int(doc_id) for doc_id, _, _ in candidate_docs]
        id_list_str = ", ".join(map(str, candidate_ids))

        # (2) Milvus 내 문서 중 Elastic 후보군 id만 필터링
        results = milvus_vectorstore.similarity_search_with_score(
            query,
            k=15,
            expr=f"id in [{id_list_str}]"
        )

        # Elastic 점수 맵 생성
        elastic_score_map = {int(doc_id): es_score for doc_id, _, es_score in candidate_docs}

        # (3) Milvus 점수와 Elastic 점수 결합
        combined_scores = []
        for doc, milvus_score in results:
            doc_id = int(doc.metadata.get("id", -1))
            elastic_score = elastic_score_map.get(doc_id, 0.0)
            final_score = milvus_score * 100 + elastic_score
            combined_scores.append((doc, final_score))

    # (4) 최종 점수 기준으로 상위 5개 문서 선택
    combined_scores.sort(key=lambda x: x[1], reverse=True)
    top_results = combined_scores[:5]

    print(f"\n[최종 결합 점수 기준 결과] 상위 {len(top_results)}개 문서:")
    for i, (doc, score) in enumerate(top_results, start=1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"\n- 문서 {i} -")
        print(f"내용(일부): {preview}...")
        print(f"결합 점수 (Milvus*100 + Elastic): {score:.4f}")
        print(f"출처: {doc.metadata.get('source', 'N/A')}")

    retrieved_docs = [doc for doc, _ in top_results]
    return {"context": retrieved_docs}

# context와 question을 가지고 답변 생성
def generate_node(state: Graph_State) -> dict:
    llm = state["llm"]
    question = state["question"]

    if state.get("skip_retrieval", False):    # 검색 건너뜀
        messages = [
            SystemMessage(content="다음 사용자의 질문에 간단히 한국어로 답변해주세요."),
            HumanMessage(content=question)
        ]
        print("\n[직접 답변 생성 중 (검색 생략)]")
        response = llm.invoke(messages)
        answer = response.content.strip()
        print(f"[응답] {answer}")
        return {"answer": answer}

    context_docs = state.get("context", [])
    context_text = "\n\n".join([doc.page_content for doc in context_docs])

    messages = [
        SystemMessage(content="""
                      당신은 반려견 전문 수의사 챗봇입니다. 아래 지침 단계에 맞게 행동해 주세요.
                      [지침]
                      1. 사용자의 질문을 읽고 이해합니다.
                      2. 제공된 문서에서 질문을 대답하는 데 도움이 되는 정보를 찾습니다.
                      3. 사용자의 질문에 항상 한국어로 친절하게 답변합니다.
                      """),
        HumanMessage(content=f"[질문] {question}\n\n[참고 문서]\n{context_text}")
    ]

    print("\n[RAG 기반 답변 생성 중...]")
    response = llm.invoke(messages)
    answer = response.content.strip()

    return {"answer": answer}

def check_quality_node(state: Graph_State) -> dict:
    return {"quality": "good"}

# 최종 답변 출력 후 종료 or 재시작
def output_node(state: Graph_State) -> dict:
    answer = state["answer"]
    print(f"[응답] {answer}")
    pass

def fallback_node(state: Graph_State) -> dict:
    pass

# 워크플로우 정의 (MemorySaver 사용하기)
def workflow():
    graph = StateGraph(Graph_State)

    graph.add_node("init_node", init_node)
    graph.add_node("input_node", input_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("check_quality_node", check_quality_node)
    graph.add_node("output_node", output_node)
    graph.add_node("fallback_node", fallback_node)

    graph.set_entry_point("init_node")

    graph.add_edge("init_node", "input_node")

    def dedicate_quit(state: Graph_State) -> bool:
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
    graph.add_edge("generate_node", "check_quality_node")

    def dedicate_review(state: Graph_State) -> str:
      if state["quality"] == "good":
          return "GOOD"
      elif state["quality"] == "bad":
          return "BAD"
      else:
          if state["retry_cnt"] >= 1:
              print("재시도 횟수 초과")
              return "FAIL"

    graph.add_conditional_edges("check_quality_node",
                                dedicate_review,
                                  {
                                    "GOOD": "output_node",
                                    "BAD": "generate_node",
                                    "FAIL": "fallback_node"
                                  }
                                )
    graph.add_edge("output_node", "input_node")
    graph.add_edge("fallback_node", "input_node")

    return graph.compile()

# PNG 이미지 생성
def draw_workflow(app):
    try:
        png_data = app.get_graph().draw_mermaid_png()
        with open("my_rag_workflow.png", "wb") as f:
            f.write(png_data)

        print("그래프가 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"이미지 생성 중 오류가 발생했습니다: {e}")

def main():
    app = workflow()
    # draw_workflow(app)

    initial_state = {}
    app.invoke(initial_state)

if __name__ == "__main__":
    main()