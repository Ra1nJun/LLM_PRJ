from langchain_milvus import Milvus
from typing import TypedDict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.retrievers import BaseRetriever
import os
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings


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
    }

  # 임베딩 모델 로드
  try:
    embedding_model = HuggingFaceEmbeddings(
      model_name=EMBEDDING_MODEL_NAME,
      model_kwargs={
          'device': 'cpu' 
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
    milvus = Milvus(
      embedding_function=embedding_model,
      collection_name=MILVUS_COLLECTION_NAME,
      connection_args={"host": MILVUS_HOST, "port": MILVUS_PORT},
      vector_field="embedding" # 벡터 필드 이름
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
def retriever_node(state: Graph_State) -> dict:
  milvus_vectorstore = state["milvus"]

  retriever = milvus_vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
  )
  retrieved_docs = retriever.invoke(state["question"])

  for i, doc in enumerate(retrieved_docs):
      print(f"\n- 문서 {i+1} -")
      print(f"내용(일부): {doc.page_content[:150]}...")

  return {"context": retrieved_docs}

# context 품질 체크
def check_context_node(context: List[Document]):
  pass

def question_rewrite_node(state):
  pass

# context와 question을 가지고 답변 생성
def generate_node(state):
  load_dotenv()

  groq_api_key=os.getenv("GROQ_API_KEY")
  model=os.getenv("GROQ_API_MODEL","openai/gpt-oss-20b")
  temp=os.getenv("GROQ_API_TEMPERATURE", "0.7")

  # 모델 설정
  llm = ChatGroq(model_name=model, temperature=temp, api_key=groq_api_key)

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