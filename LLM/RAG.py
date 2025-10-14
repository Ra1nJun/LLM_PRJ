from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from typing import TypedDict, List
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables.graph_mermaid import draw_mermaid_png
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

# 상태 정의
class Graph_State(TypedDict):
  questuion: str
  context: List[Document]
  answer: str

# 데이터 초기화
def init(state):
  pass

# 사용자에게 질문 받기
def input(state):
  pass

# 벡터DB에서 유사도 검색
def retriever(state):
  pass

# context와 question을 가지고 답변 생성
def generate(state):
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
def output(state):
  pass

# 워크플로우 정의 (MemorySaver 사용하기)
def workflow():
  graph = StateGraph(Graph_State)

  graph.add_node("init", init)
  graph.add_node("input", input)
  graph.add_node("retriever", retriever)
  graph.add_node("generate", generate)
  graph.add_node("output", output)

  graph.set_entry_point("init")

  graph.add_edge("init", "input")
  graph.add_edge("input", "retriever")
  graph.add_edge("retriever", "generate")
  graph.add_edge("generate", "output")

  def dedicate_restart(state: Graph_State) -> bool:
    if state["question"].lower().strip() in ["exit", "quit", "q", "종료", "끝", "그만"]:
      return "end"
    else:
      return "input" 

  graph.add_conditional_edges("output",
                              dedicate_restart,
                                {
                                  "input": "input",
                                  "end" : END
                                }
                              )
  
  return graph.compile()

def main():
  app = workflow()
  try:
    mermaid_code = app.get_graph().draw_mermaid()   # Mermaid 코드 뽑기
    png_data = draw_mermaid_png(mermaid_code)       # PNG 이미지 생성

    with open("my_rag_workflow.png", "wb") as f:
        f.write(png_data)

    print("그래프가 성공적으로 저장되었습니다.")
  except Exception as e:
    print(f"이미지 생성 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
  main()