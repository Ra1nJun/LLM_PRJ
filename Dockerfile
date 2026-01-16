FROM docker.elastic.co/elasticsearch/elasticsearch:9.1.10

# nori 플러그인 설치
RUN elasticsearch-plugin install --batch analysis-nori