from dotenv import load_dotenv
import requests
import os

load_dotenv()

API_KEY = os.getenv("S2ORC_API_KEY") or "NotFound"
HEADERS = {"x-api-key": API_KEY}
RELEASE = "2026-10-27"
DOWNLOAD_DIR = f"s2orc_shards-{RELEASE}"

def download_all_s2orc_shards():
    if (API_KEY == "NotFound"):
        print("No API Key provided.")
        return False
    
    response = requests.get(f"https://api.semanticscholar.org/datasets/v1/release/{RELEASE}/dataset/s2orc_v2", headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Authentication Failed: {response.status_code}\n{response.text}")
        return False

    files = response.json().get("files", [])
    num_files = len(files)
    print(f"Found {num_files} shards.")
    
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for i, shard_url in enumerate(files):
        file_name = f"s2orc_v2_shard_{i:03d}.gz"
        file_path = os.path.join(DOWNLOAD_DIR, file_name)
        
        if os.path.exists(file_path):
            print(f"Skipping {file_name} - already downloaded.")
            continue
            
        with requests.get(shard_url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                    f.write(chunk)
                    
        print(f"Downloaded {file_name}")
        
    print("\nAll shards downloaded!")