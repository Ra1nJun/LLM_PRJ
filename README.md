# LLM_PRJ
Pet Dog Disease & Growth Chatbot

## Overview
해당 프로젝트는 반려 강아지의 성장, 질병 등 생애주기에 따른 필요한 정보를 답해주는 챗봇 사이트이다.

## 주요 기능
 - 사용자에 질의에 응답해주는 RAG 기반 LLM
 - 자체적인 사용자의 계정 관리

## 설정 및 실행
 1. env와 같은 개인 설정
 2. mariaDB에 계정 생성 및 권한 부여
 3. docker-compose up 실행
 4. LLM/Indexing 을 이용해 데이터(preprocessing.py 사용된 데이터) 임베딩 후 저장
 5. mkcert 설치하여 pem 생성
 6. https로 BACKEND와 FRONTEND 실행

## 기술 스택
|Front|Back|DB|CI|
|:---:|:---:|:---:|:---:|
|react / JS|Python / FastAPI|ElasticSearch / Milvus / MariaDB|docker / GitHub Actions|


## 폴더 구조
```
LLM-PRJ
+---BACKEND
|   +---app
|       +---api
|           +---endpoints
|           +---routers.py
|       +---core
|       +---crud
|       +---db
|       +---models
|       +---schemas
+---FRONTEND
|   +---src
|       +---api
|       +---assets
|       +---components
|       +---context
|       +---hooks
|       +---pages
+---LLM
|   +---Agent.py (Lang Graph 워크플로우 RAG)
|   +---Indexing.py (데이터 임베딩 저장)
+---DATA
|   +---...
+---.env
+---.pem
+---preprocessing.py
```

## Git Branch 전략
|브랜치명|설명|
|:---:|:---|
|main|안정적인 버전|
|dev|다음 버전을 위한 준비|
|feat|새로운 기능|
|hotfix|긴급 버그 수정|
|refactoring|구조 단순화 및 수정|
> 예시: feat/LLM/simple_llm

## 데이터 출처
 - 반려견 성장 및 질병관련 말뭉치 데이터 : https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=data&dataSetSn=71879