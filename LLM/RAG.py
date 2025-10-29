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


load_dotenv()
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "my_collection")
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")

# 상태 정의
class Graph_State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    context_quality: str
    rewrite_count: int
    milvus: Optional[Milvus]
    embedding_model: Optional[Embeddings]
    llm: Optional[BaseLLM]

# 실행 준비
def init_node(state: Graph_State) -> dict:
    # 데이터 초기화
    initial_state = {
        "question": "",
        "context": [],
        "answer": "",
        "context_quality": "",
        "rewrite_count": 0,
        "milvus": None,
        "embedding_model": None,
        "llm": None,
    }

    # 임베딩 모델 로드
    try:
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

    # LLM 모델 로드
    try:
        load_dotenv()

        groq_api_key=os.getenv("GROQ_API_KEY")
        model=os.getenv("GROQ_API_MODEL","openai/gpt-oss-20b")
        temp=os.getenv("GROQ_API_TEMPERATURE", "0.7")

        # 모델 설정
        llm = ChatGroq(model_name=model, temperature=temp, api_key=groq_api_key)
        initial_state["llm"] = llm
        print(f"LLM 모델({model}) 로드 성공")
    except Exception as e:
        print(f"LLM 모델({model}) 로드 실패: {e}")
        exit(1)

    # MilvusDB 연결
    try:
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

    return initial_state

# 사용자에게 질문 받기
def input_node(state: Graph_State) -> dict:
    user_input = input("증상을 입력하세요(종료하고 싶다면 quit): ")
    return {"question": user_input}

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
            hit["_source"].get("text", "")
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
    # candidate_docs = []

    if not candidate_docs:
        print("[경고] ElasticSearch에서 후보군을 찾지 못했습니다. → Milvus만으로 검색 수행")
        results = milvus_vectorstore.similarity_search_with_score(query, k=15)
    else:
        candidate_ids = [int(doc_id) for doc_id, _ in candidate_docs]
        id_list_str = ", ".join(map(str, candidate_ids))

        # (2) Milvus 내 문서 중 Elastic 후보군 id만 필터링
        results = milvus_vectorstore.similarity_search_with_score(
            query,
            k=15,
            expr=f"id in [{id_list_str}]"
        )

    # (3) 결과 출력
    print(f"\n[최종 결과] 상위 {len(results)}개 문서:")
    for i, (doc, score) in enumerate(results, start=1):
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"\n- 문서 {i} -")
        print(f"내용(일부): {preview}...")
        print(f"유사도 점수: {score:.4f}")
        print(f"출처: {doc.metadata.get('source', 'N/A')}")

    retrieved_docs = [doc for doc, _ in results]
    return {"context": retrieved_docs}

# context 품질 체크
def check_context_node(state: Graph_State) -> dict:
  llm = state["llm"]
  context = state["context"]
  question = state["question"]
  messages = [
  SystemMessage(content=
                  """
                  당신은 반려동물 전문 수의사의 보조입니다. 다음은 반려동물의 증상에 대한 질문과 관련 문서들입니다.
                  관련 문서들이 질문에 적합한 정보를 포함하고 있는지 평가해주세요.
                  """
                )
  ]

def question_rewrite_node(state: Graph_State) -> dict:
  pass

# context와 question을 가지고 답변 생성
def generate_node(state: Graph_State) -> dict:
  # 메시지 설정 (1. 목적(역할), 2. 받을 변수, 3. 제약 조건, 4. 출력시 요구 사항)
  messages = [
  SystemMessage(content=
                  """
                  """
                )
  ]

  messages.append(HumanMessage(content=state["questuion"]))

  state["answer"]= llm.invoke(messages)
  answer = state["answer"].content

  return answer

# 최종 답변 출력 후 종료 or 재시작
def output_node(state):
  pass

# 워크플로우 정의 (MemorySaver 사용하기)
def workflow():
  graph = StateGraph(Graph_State)

  graph.add_node("init_node", init_node)
  graph.add_node("input_node", input_node)
  graph.add_node("retriever_node", retriever_node)
  graph.add_node("check_context_node", check_context_node)
  graph.add_node("question_rewrite_node", question_rewrite_node)
  graph.add_node("generate_node", generate_node)
  graph.add_node("output_node", output_node)

  graph.set_entry_point("init_node")

  graph.add_edge("init_node", "input_node")

  def dedicate_quit(state: Graph_State) -> bool:
    if state["question"].lower() == "quit":
      print("프로그램을 종료합니다.")
      return True
    else:
      return False

  graph.add_conditional_edges("input_node",
                              dedicate_quit,
                                {
                                  False: "retriever_node",
                                  True : END
                                }
                              )
  graph.add_edge("retriever_node", "check_context_node")

  def dedicate_review(state: Graph_State) -> str:
    match(state["context_quality"]):
      case "good":
        return "generate_node"
      case "bad":
        return "question_rewrite_node"
      case _:
        return "output_node"

  graph.add_conditional_edges("check_context_node",
                              dedicate_review,
                                {
                                  "generate_node": "generate_node",
                                  "question_rewrite_node": "question_rewrite_node",
                                  "output_node": "output_node"
                                }
                              )
  graph.add_edge("question_rewrite_node", "retriever_node")
  graph.add_edge("generate_node", "output_node")
  graph.add_edge("output_node", "input_node")

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