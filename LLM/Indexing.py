import os
import fitz
import logging
from functools import lru_cache
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from FlagEmbedding import BGEM3FlagModel

# -------------------------------
# 설정(Config)
# -------------------------------
CONFIG = {
    "pdf_path": "./DATA",
    "embedding_model_name": "BAAI/bge-m3",
    "chunk_size": 1200,
    "chunk_overlap": 200,
    "milvus_host": "localhost",
    "milvus_port": "19530",
    "collection_name": "my_collection",
    "batch_size": 16,
}

# -------------------------------
# 로깅 설정
# -------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# Milvus 핸들러 클래스
# -------------------------------
class MilvusHandler:
    def __init__(self, host="localhost", port="19530", collection_name="my_collection"):
        self.client = MilvusClient(host=host, port=port)
        self.collection_name = collection_name
        self._setup_collection()

    def _setup_collection(self):
        if self.client.has_collection(self.collection_name):
            logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제 중...")
            self.client.drop_collection(self.collection_name)

        schema = CollectionSchema(
            auto_id=True,
            enable_dynamic_field=False,
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=4096),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=128),
            ],
        )
        self.client.create_collection(collection_name=self.collection_name, schema=schema)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="embedding", index_type="FLAT", metric_type="L2")
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)
        logger.info(f"Milvus 컬렉션 '{self.collection_name}' 생성 완료.")

    def insert(self, embeddings, texts, source):
        entities = [
            {"embedding": emb, "text": txt, "source": source}
            for emb, txt in zip(embeddings, texts)
        ]
        self.client.insert(self.collection_name, entities)

    def load(self):
        self.client.load_collection(self.collection_name)
        logger.info(f"컬렉션 '{self.collection_name}' 로드 완료.")

# -------------------------------
# PDF 관련 유틸 함수
# -------------------------------
def extract_blocks_from_pdf(pdf_path):
    """PDF에서 텍스트 블록을 추출하여 리스트로 반환"""
    text_blocks = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text = block[4].strip()
                    if text:
                        text_blocks.append(text)
    except Exception as e:
        logger.error(f"PDF 추출 실패 ({pdf_path}): {e}")
    return text_blocks

def create_chunks_with_overlap(text_blocks, chunk_size, chunk_overlap):
    """블록을 합쳐 슬라이딩 윈도우 방식으로 청크 생성"""
    combined_text = "\n".join(text_blocks)
    chunks = []
    start = 0
    text_length = len(combined_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(combined_text[start:end])
        start += chunk_size - chunk_overlap

    return chunks

def get_pdf_files(directory):
    """디렉토리 내 PDF 파일 목록 반환"""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".pdf")
    ]

# -------------------------------
# 임베딩 관련 함수
# -------------------------------
@lru_cache(maxsize=1)
def get_embedding_model(model_name):
    """임베딩 모델 캐싱 로드"""
    logger.info(f"임베딩 모델 '{model_name}' 로드 중...")
    return BGEM3FlagModel(model_name, use_fp16=True)

def batch_embedding(chunks, embedding_model, batch_size):
    """텍스트 청크를 배치 단위로 임베딩"""
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            batch_embeddings = embedding_model.encode(batch, return_dense=True)
            embeddings.extend(batch_embeddings["dense_vecs"])
        except Exception as e:
            logger.error(f"임베딩 실패 (batch {i}): {e}")
    return embeddings

# -------------------------------
# 단일 PDF 파일 처리 함수
# -------------------------------
def process_pdf_file(pdf_file_path, embedding_model, milvus_handler, config):
    source_name = os.path.basename(pdf_file_path)
    logger.info(f"\n[INFO] 현재 처리 파일 : {source_name}")

    text_blocks = extract_blocks_from_pdf(pdf_file_path)
    if not text_blocks:
        logger.warning(f"'{source_name}'에서 텍스트 블록 추출 실패.")
        return

    text_chunks = create_chunks_with_overlap(
        text_blocks,
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"],
    )

    if not text_chunks:
        logger.warning(f"'{source_name}'에서 유효한 텍스트 청크가 없음.")
        return

    embeddings = batch_embedding(
        text_chunks,
        embedding_model,
        batch_size=config["batch_size"],
    )

    milvus_handler.insert(embeddings, text_chunks, source_name)
    logger.info(f"'{source_name}' → {len(text_chunks)}개 청크 삽입 완료.")

# -------------------------------
# main 실행 함수
# -------------------------------
def main():
    config = CONFIG

    logger.info("Milvus 연결 및 컬렉션 생성 중...")
    try:
        milvus_handler = MilvusHandler(
            host=config["milvus_host"],
            port=config["milvus_port"],
            collection_name=config["collection_name"],
        )
    except Exception as e:
        logger.error(f"Milvus 연결 실패: {e}")
        return

    embedding_model = get_embedding_model(config["embedding_model_name"])

    pdf_files = get_pdf_files(config["pdf_path"])
    if not pdf_files:
        logger.warning(f"'{config['pdf_path']}' 디렉토리에 PDF 파일이 없습니다.")
        return

    logger.info(f"총 {len(pdf_files)}개 PDF 파일 확인됨.")
    for pdf_file_path in pdf_files:
        process_pdf_file(pdf_file_path, embedding_model, milvus_handler, config)

    milvus_handler.load()
    logger.info("\n✅ 인덱싱 작업 완료!")


if __name__ == "__main__":
    main()