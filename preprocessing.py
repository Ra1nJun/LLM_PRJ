import os
import json
import re

folder_path = "./DATA"

for root, dirs, files in os.walk(folder_path):
    for file_name in files:
        if file_name.lower().endswith(".json"):
            file_path = os.path.join(root, file_name)
            print(f"처리 중: {file_path}")

            # BOM 제거하며 읽기
            with open(file_path, "r", encoding="utf-8-sig") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    print(f"⚠ JSON 로드 실패: {file_path} -> {e}")
                    continue

            # "disease" 키가 있으면 연속 \n 처리
            if "disease" in data and isinstance(data["disease"], str):
                data["disease"] = re.sub(r'\n{2,}', '\n', data["disease"])

            # BOM 없는 UTF-8로 다시 저장
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

print("모든 JSON 파일 처리 완료!")