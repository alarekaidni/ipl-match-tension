import os
import urllib.request
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

URLS = {
    "matches.csv": "https://raw.githubusercontent.com/shivam-gupta0/IPL-Data-Analysis/master/matches.csv",
    "deliveries.csv": "https://raw.githubusercontent.com/shivam-gupta0/IPL-Data-Analysis/master/deliveries.csv"
}

def download_file(filename, url):
    target_path = os.path.join(DATA_DIR, filename)
    if os.path.exists(target_path) and os.path.getsize(target_path) > 1000:
        print(f"[ALREADY EXISTS] {filename} ({os.path.getsize(target_path):,} bytes)")
        return
    print(f"[DOWNLOADING] {filename} from {url}...")
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = downloaded / total_size * 100
            sys.stdout.write(f"\r  -> {percent:.1f}% ({downloaded / 1e6:.2f} MB / {total_size / 1e6:.2f} MB)")
            sys.stdout.flush()
    urllib.request.urlretrieve(url, target_path, reporthook)
    print(f"\n[DONE] Saved to {target_path} ({os.path.getsize(target_path):,} bytes)")

if __name__ == "__main__":
    for fname, url in URLS.items():
        download_file(fname, url)
