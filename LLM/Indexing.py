from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import fitz
import torch
from langdetect import detect
import os

pdf_path = "./DATA"
embedding_model_name = "BAAI/bge-m3"

# MilvusDB 연결 확인 및 컬렉션 생성
def check_milvus_and_create():
  VectorDB = MilvusClient(host='localhost', port='19530')

  if VectorDB.has_collection('my_collection'):
    VectorDB.drop_collection('my_collection')

  custom_schema = CollectionSchema(auto_id=True, enable_dynamic_field=False,
                                    fields=[
                                      FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                                      FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
                                      FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=1024),
                                      FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=128)
                                    ]
  )
  VectorDB.create_collection(collection_name='my_collection', schema=custom_schema)
  return VectorDB

# PDF에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
  doc = fitz.open(pdf_path)
  text_chunk = []
  for page in doc:
    texts = page.get_text('blocks') # blocks는 문단 단위로 텍스트를 추출
    for block in texts:
      text_chunk.append(block[4].strip())
  doc.close()

  return [chunk for chunk in text_chunk if chunk]

def batch_translate(chunks, batch_size=16):
  model_name = "Helsinki-NLP/opus-mt-en-ko"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  translated_chunks = []

  for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]

    # 모델 입력 형태로 변환 및 번역
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True)
    with torch.no_grad():
      translated = model.generate(**tokenized_batch)

    # 번역된 토큰을 다시 텍스트로 디코딩
    decoded_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
    translated_chunks.extend(decoded_batch)

  return translated_chunks

# batch 임베딩
def batch_embedding(chunks, embedding_model, batch_size=32):
  embeddings = []

  for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_embeddings = embedding_model.encode(batch, return_dense=True)
    embeddings.extend(batch_embeddings['dense_vecs'])
  return embeddings

def insert_to_milvus(embeddings, chunks, source, VectorDB):
  entities = []

  for i in range(len(chunks)):
    entity={
      'embedding': embeddings[i],
      'text': chunks[i],
      'source': source
    }
    entities.append(entity)
  
  MilvusClient.insert('my_collection', entities)

def get_pdf_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

def main():
  global pdf_path, embedding_model_name

  print("MilvusDB 연결 확인 및 컬렉션 생성 중...")
  VectorDB = check_milvus_and_create()

  print("임베딩 모델 로드 중...")
  embedding_model = BGEM3FlagModel(embedding_model_name, use_fp16=True)

  file_list = get_pdf_files(pdf_path)
  if not file_list:
    print("No PDF files found in the specified directory.")
    return

  print(f"총 {len(file_list)}개의 PDF 파일 확인")
  for pdf_path in file_list:
    source_name = os.path.basename(pdf_path)
    print(f"\n처리하고 있는 파일: {source_name}")

    text_chunks = extract_text_from_pdf(pdf_path)

    if not text_chunks:
        print(f"'{source_name}' 파일에서 텍스트 감지 실패")
        continue

    try:
        lang = detect(text_chunks[0])
        print(f"{lang} 언어 감지")
    except:
        lang = 'unknown'
        print(f"'{source_name}' 에서 언어 감지 실패")
        continue

    processed_chunks = []

    if lang == 'ko':
      print("한국어 감지로 번역 없이 임베딩 진행")
      processed_chunks = text_chunks
    elif lang == 'en':
      print("영어 감지로 번역 후 임베딩 진행")
      processed_chunks = batch_translate(text_chunks, batch_size=16)
    else:
        print(f"지원하지 않는 언어로 '{source_name}' 파일 건너뜀")
        continue

    embeddings = batch_embedding(processed_chunks, embedding_model, batch_size=32)
    insert_to_milvus(embeddings, processed_chunks, source_name, VectorDB)
  
  print("\n인덱싱 작업 완료")

if __name__ == "__main__":
    main()