from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key=os.getenv("GROQ_API_KEY")
model=os.getenv("GROQ_API_MODEL")
temp=os.getenv("GROQ_API_TEMPERATURE")

# 모델 설정
llm = ChatGroq(model_name=model, temperature=temp, api_key=groq_api_key)

# 메시지 설정
messages = [
  SystemMessage(content=
                  """
                  당신은 llm의 모든 것을 알고 있는 전문가입니다.
                  사용자의 질문이나 요구사항에는 단계적으로 차근차근 아주 자세하게 초등학생도 이해시킬 수 있을 정도로 답변해야 합니다.
                  또한, 사용자와 일상적인 대화도 아주 섬세하게 답변합니다.
                  한국어로만 대답해야 합니다.
                  """
                )
]

# 대화형으로 계속 대화
while True:
  try:
    user_input = input("\n입력(종료를 원하면 exit): ")
    if user_input.lower() == "exit":
      print("대화 종료")
      break
    
    # 사용자 입력 저장
    messages.append(HumanMessage(content=user_input))

    # 메시지(이전 대화 리스트 전부) 전달
    response = llm.invoke(messages)

    # 모델의 응답 저장
    messages.append(AIMessage(content=response.content))
    print(f"--------------- GroqAI ---------------\n{response.content}\n --------------------------------------")
  except Exception as e:
    print(f"오류 발생: {e}")
    break