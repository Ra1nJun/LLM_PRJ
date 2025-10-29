import os
import json
import logging
from functools import lru_cache
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from FlagEmbedding import BGEM3FlagModel
from elasticsearch import Elasticsearch, helpers
import itertools
import re

# -------------------------------
# 설정
# -------------------------------
CONFIG = {
    "data_path": "./DATA",
    "embedding_model_name": "BAAI/bge-m3",
    "milvus_host": "localhost",
    "milvus_port": "19530",
    "collection_name": "my_collection",
    "es_host": "localhost",
    "es_port": 9200,
    "es_index": "my_index",
    "batch_size": 16,
    "source_field": "title",
    "text_field": "disease",
    "department_field": "department",
    "start_id": 1,
    "max_bytes": 1024
}

# -------------------------------
# 로깅 설정
# -------------------------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# -------------------------------
# Milvus, elastic 클래스
# -------------------------------
class DBHandler:
    def __init__(self, host="localhost", port="19530", collection_name="my_collection", es_host="localhost", es_port=9200, es_index="my_index"):
        self.client = MilvusClient(host=host, port=port)
        self.collection_name = collection_name
        
        # Elasticsearch 설정
        self.es = Elasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])
        self.es_index = es_index

        if self.es.indices.exists(index=self.es_index):
            self.es.indices.delete(index=self.es_index)
            logger.info(f"기존 Elasticsearch 인덱스 '{self.es_index}' 삭제 완료.")

        index_config = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "korean_analyzer": {
                            "type": "custom",
                            "tokenizer": "nori_tokenizer",
                            "filter": ["lowercase", "nori_part_of_speech"]
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "text": {
                        "type": "text",
                        "analyzer": "korean_analyzer"
                    }
                }
            }
        }

        self.es.indices.create(index=self.es_index, body=index_config)
        logger.info(f"Elasticsearch 인덱스 '{self.es_index}' 생성 완료.")
        
        self.id_counter = itertools.count(CONFIG["start_id"])
        self._setup_collection()

    def _setup_collection(self):
        if self.client.has_collection(self.collection_name):
            logger.info(f"기존 컬렉션 '{self.collection_name}' 삭제 중...")
            self.client.drop_collection(self.collection_name)

        schema = CollectionSchema(
            auto_id=False,
            enable_dynamic_field=False,
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=16)
            ],
        )
        self.client.create_collection(collection_name=self.collection_name, schema=schema)
        index_params = self.client.prepare_index_params()
        index_params.add_index(field_name="vector", index_type="FLAT", metric_type="COSINE")
        self.client.create_index(collection_name=self.collection_name, index_params=index_params)
        logger.info(f"Milvus 컬렉션 '{self.collection_name}' 생성 완료.")

    def insert(self, vectors, texts, title, department):
        batch_size = CONFIG["batch_size"] 
        total = len(vectors)
        
        for i in range(0, total, batch_size):
            end = i + batch_size
            batch_vectors = vectors[i:end]
            batch_texts = texts[i:end]

            # 엔티티
            entities = []
            for v, t in zip(batch_vectors, batch_texts):
                entities.append({"id": next(self.id_counter), "vector": v, "text": t, "source": title, "department": department})

            try:
                self.client.insert(self.collection_name, entities)
            except Exception as e:
                logger.warning(f"텍스트 길이 초과 내용 : {t[:200]}...")
                logger.warning(f"bytes : {len(t.encode('utf-8'))} bytes")
                logger.error(f"Milvus 배치 삽입 실패 ({i}-{end}): {e}")
                continue

            # Elasticsearch용 문서
            actions = [
                {
                    "_index": self.es_index,
                    "_id": e["id"],  # Milvus auto_id를 Elasticsearch _id로 사용
                    "_source": {
                        "text": e["text"]
                    }
                }
                for e in entities
            ]
            try:
                helpers.bulk(self.es, actions)
                logger.info(f"Elasticsearch에 {len(actions)}개 문서 삽입 완료.")
            except Exception as e:
                logger.error(f"Elasticsearch 삽입 실패 ({i}-{end}): {e}")

    def load(self):
        self.client.load_collection(self.collection_name)
        logger.info(f"컬렉션 '{self.collection_name}' 로드 완료.")

# -------------------------------
# Embedding 클래스
# -------------------------------
class Embedding:
    def __init__(self, model_name):
        self.model = self._load_model(model_name)

    @staticmethod    # 클래스 상태와 독립적인 함수
    @lru_cache(maxsize=1)    # 같은 모델을 여러 객체에서 반복해서 호출하더라도 캐시된 모델을 재사용할 수 있도록 함
    def _load_model(model_name):    # 실제 모델 불러오는 함수
        logger.info(f"GPU 기반으로 임베딩 모델 '{model_name}' 로드 중...")
        return BGEM3FlagModel(model_name, use_fp16=True, device='cuda')

    def embed_batches(self, chunks, batch_size):    # 배치 단위로 임베딩을 수행하는 함수
        vectors = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            try:
                batch_embeddings = self.model.encode(batch, return_dense=True)
                vectors.extend(batch_embeddings["dense_vecs"].astype("float32"))
            except Exception as e:
                logger.error(f"GPU 임베딩 실패 (batch {i}): {e}")
        return vectors

# -------------------------------
# DataProcess 클래스
# -------------------------------
class DataProcess:
    def __init__(self, config):
        self.config = config

    def find_json(self, directory):    # 경로 내의 모든 json 파일 탐색
        json_files = []
        logger.debug(f"'{directory}'에서 json 파일 탐색 중...")
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(".json"):
                    json_files.append(os.path.join(root, f))
        return json_files

    def load_json(self, json_file_path):
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            text_field = self.config["text_field"]
            source_field = self.config["source_field"]
            department_field = self.config["department_field"]

            combined_text = str(data.get(text_field, "")).strip()
            title = str(data.get(source_field, os.path.basename(json_file_path))).strip()
            department = str(data.get(department_field, "N/A")).strip()

        except Exception as e:
            logger.error(f"json 로드 실패 ({json_file_path}): {e}")
            return None, None, []

        if not combined_text:
            logger.warning(f"'{text_field}' 필드가 비어있음: {json_file_path}")
            return None, None, []

        # 문장 분리 (. 또는 개행 기준)
        sentences = re.split(
            r'(?<=[가-힣\)\.])\.\s+|(?<=[a-zA-Z\)\]])\.\s+(?=[A-Z])',
            combined_text
        )
        sentences = [s.strip() for s in sentences if s.strip()]

        # 2048 bytes 기준 병합 (문장 단위)
        chunks = []
        current_chunk = ""
        max_bytes = self.config["max_bytes"]

        for sentence in sentences:
            sentence_bytes = len(sentence.encode("utf-8"))

            # 현재 청크에 합치면 초과하는지 체크
            if len(current_chunk.encode("utf-8")) + sentence_bytes + 1 <= max_bytes:
                current_chunk = f"{current_chunk} {sentence}".strip() if current_chunk else sentence
            else:
                # 현재 청크를 저장하고 새 청크 시작
                if current_chunk:
                    safe_chunk = current_chunk.encode("utf-8")[:max_bytes].decode("utf-8", errors="ignore")
                    chunks.append(safe_chunk)
                current_chunk = sentence

        # 마지막 청크 추가
        if current_chunk:
            chunks.append(current_chunk)

        return title, department, chunks

    def process_json(self, json_file_path, embedder, milvus_handler):    # load, embed, insert 함수들을 호출하여 json 데이터를 처리
        file_name = os.path.basename(json_file_path)
        logger.debug(f"현재 처리 파일 : {file_name}")
        title, department, text_chunks = self.load_json(json_file_path)
        
        if not text_chunks or not title or not department:
            logger.warning(f"'{file_name}' 처리 실패.")
            return
        
        batch_size = self.config["batch_size"]
        vectors = embedder.embed_batches(text_chunks, batch_size=batch_size)
        
        if len(vectors) != len(text_chunks):
            logger.error(f"'{file_name}' 임베딩 수({len(vectors)})와 청크 수({len(text_chunks)}) 불일치 → 삽입 중단.")
            return

        milvus_handler.insert(vectors, text_chunks, title, department)
        logger.debug(f"{len(text_chunks)}개 청크 삽입 완료.")

# -------------------------------
# main 함수
# -------------------------------
def main():
    config = CONFIG

    milvus_handler = DBHandler(
        host=config["milvus_host"],
        port=config["milvus_port"],
        collection_name=config["collection_name"],
        es_host=config["es_host"],
        es_port=config["es_port"],
        es_index=config["es_index"]
    )
    embedder = Embedding(config["embedding_model_name"])
    data_processor = DataProcess(config)

    json_files = data_processor.find_json(data_processor.config["data_path"]) 
    
    if not json_files:
        logger.warning(f"'{config['data_path']}' 내 json 파일이 없습니다.")
        return
    logger.info(f"총 {len(json_files)}개 json 파일 확인됨.")

    for json_file_path in json_files:
        data_processor.process_json(json_file_path, embedder, milvus_handler)

    milvus_handler.load()
    logger.info("\n인덱싱 완료!")

if __name__ == "__main__":
    main()