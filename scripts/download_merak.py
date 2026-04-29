import os
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    from icrawler.builtin import GoogleImageCrawler
except ImportError:
    print("icrawler not found, installing...")
    install("icrawler")
    from icrawler.builtin import GoogleImageCrawler

query = "batik priangan merak ngibing"
base_dataset_dir = r"c:\Users\Benny Pepper\Documents\GitHub\CNN-Batik-Motifs-Detector\v2\dataset"
target_dir = os.path.join(base_dataset_dir, "Priangan_Merak_Ngibing")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f"Searching images for query: '{query}'")

google_crawler = GoogleImageCrawler(
    feeder_threads=1,
    parser_threads=1,
    downloader_threads=4,
    storage={'root_dir': target_dir}
)

google_crawler.crawl(
    keyword=query,
    filters={'type': 'photo'},
    max_num=150,
    file_idx_offset='auto'
)

# Clean up non-jpgs
count = 0
for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    if os.path.isfile(file_path):
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            os.remove(file_path)
            print(f"Removed non-jpg file: {filename}")
        else:
            count += 1

print(f"\nSuccessfully added {count} images to {target_dir}")
