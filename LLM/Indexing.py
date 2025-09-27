from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer
import torch
from langdetect import detect
import os
import pdfplumber
from docling_core.types.doc.document import DoclingDocument

pdf_path = "./DATA"
embedding_model_name = "BAAI/bge-m3"
translate_model_name = "seongs/ke-t5-base-aihub-koen-translation-integrated-10m-en-to-ko"

# ================== MilvusDB 연결 확인 및 컬렉션 생성 ==================
def check_milvus_and_create():
  VectorDB = MilvusClient(host='localhost', port='19530')

  if VectorDB.has_collection('my_collection'):
    VectorDB.drop_collection('my_collection')

  custom_schema = CollectionSchema(auto_id=True, enable_dynamic_field=False,
                                    fields=[
                                      FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
                                      FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, dim=1024),
                                      FieldSchema(name='text', dtype=DataType.VARCHAR, max_length=2048),
                                      FieldSchema(name='source', dtype=DataType.VARCHAR, max_length=128)
                                    ]
  )
  VectorDB.create_collection(collection_name='my_collection', schema=custom_schema)

  # 1. IndexParams 객체 생성
  index_params = VectorDB.prepare_index_params()

  # 2. add_index 메서드를 사용하여 인덱스 파라미터 추가
  index_params.add_index(
      field_name='embedding',
      index_type="FLAT",
      metric_type="L2"
  )

  # 3. create_index 호출 시 IndexParams 객체 전달
  VectorDB.create_index(
      collection_name='my_collection',
      index_params=index_params
  )
  
  return VectorDB

# ================== PDF에서  추출 ==================
def extract_chunks_with_docling_semantic(pdf_path, embed_model_name):
    # 1) PDF → 텍스트 (간단 추출)
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
      for page in pdf.pages:
        text += page.extract_text() or ""

    if not text.strip():
      return []

    # 2) HuggingFace tokenizer
    hf_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
    tokenizer = HuggingFaceTokenizer(tokenizer=hf_tokenizer)

    # 3) DoclingDocument 생성
    dl_doc = DoclingDocument(
      metadata={"source": pdf_path},
      pages=[{"text": text}]
    )

    # 4) 의미적 청크 분할
    chunker = HybridChunker(
      tokenizer=tokenizer,
      max_characters=800,
      max_tokens=1024,
      split_sentences=True,
      overlap=50,
    )

    chunks = []
    for chunk in chunker.chunk(dl_doc):  # ⚡ str이 아닌 DoclingDocument 전달
      if chunk.text.strip():
        chunks.append(chunk.text.strip())

    return chunks
# ================== 배치 번역 및 임베딩 ==================
def batch_translate(chunks, batch_size=16, model_name=translate_model_name):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
  translated_chunks = []

  for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]

    # 모델 입력 형태로 변환 및 번역
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
      translated = model.generate(**tokenized_batch)

    # 번역된 토큰을 다시 텍스트로 디코딩
    decoded_batch = tokenizer.batch_decode(translated, skip_special_tokens=True)
    translated_chunks.extend(decoded_batch)

  return translated_chunks

# ================== batch 임베딩 ==================
def batch_embedding(chunks, embedding_model, batch_size=16):
  embeddings = []

  for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i+batch_size]
    batch_embeddings = embedding_model.encode(batch, return_dense=True)
    embeddings.extend(batch_embeddings['dense_vecs'])
  return embeddings

# ================== MilvusDB에 삽입 ==================
def insert_to_milvus(embeddings, chunks, source, VectorDB):
  entities = []

  for i in range(len(chunks)):
    entity={
      'embedding': embeddings[i],
      'text': chunks[i],
      'source': source
    }
    entities.append(entity)
  
  VectorDB.insert('my_collection', entities)

# ================== 메인 함수 ==================
def get_pdf_files(directory):
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

def main():
  global pdf_path, embedding_model_name, translate_model_name

  print("MilvusDB 연결 확인 및 컬렉션 생성 중...")
  try:
    VectorDB = check_milvus_and_create()
  except:
    print("MilvusDB 연결 실패. MilvusDB가 실행 중인지 확인하세요.")

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

    # 의미적 청크 분할
    text_chunks = extract_chunks_with_docling_semantic(pdf_path, embedding_model_name)
    print("청크 수:", len(text_chunks))

    if not text_chunks:
      print(f"'{source_name}' 파일에서 텍스트 감지 실패")
      continue

    try:
      lang = detect(text_chunks[0])
      print(f"{lang} 언어 감지")
    except:
      lang = 'ko'
      print(f"'{source_name}' 에서 언어 감지 실패로 기본 언어로 설정(ko)")

    processed_chunks = []

    if lang == 'ko':
      print("한국어 감지로 번역 없이 임베딩 진행")
      processed_chunks = text_chunks
    elif lang == 'en':
      print("영어 감지로 번역 후 임베딩 진행")
      processed_chunks = batch_translate(text_chunks, batch_size=8, model_name=translate_model_name)

    embeddings = batch_embedding(processed_chunks, embedding_model, batch_size=16)
    insert_to_milvus(embeddings, processed_chunks, source_name, VectorDB)
  
  VectorDB.load_collection(collection_name='my_collection')

  print("\n인덱싱 작업 완료")

if __name__ == "__main__":
    main()