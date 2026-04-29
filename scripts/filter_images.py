import os
import shutil

target_dir = r"C:\Users\Benny Pepper\Downloads\Batik_dataset\merak_nigbing_downloaded"
valid_exts = {".jpg", ".jpeg", ".png", ".webp"}

print(f"Scanning directory: {target_dir}")

files_info = []

for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    if not os.path.isfile(file_path):
        continue
        
    ext = os.path.splitext(filename)[1].lower()
    
    # Delete non-images
    if ext not in valid_exts:
        os.remove(file_path)
        print(f"Deleted non-image file: {filename}")
        continue
        
    # Get file size
    size_kb = os.path.getsize(file_path) / 1024
    
    # Delete tiny thumbnails (less than 20 KB)
    if size_kb < 20:
        os.remove(file_path)
        print(f"Deleted tiny thumbnail: {filename} ({size_kb:.1f} KB)")
        continue
        
    files_info.append((file_path, filename, size_kb))

# Sort by size descending (largest/highest quality first)
files_info.sort(key=lambda x: x[2], reverse=True)

# Keep top 30, delete the rest
KEEP_COUNT = 30
kept_files = files_info[:KEEP_COUNT]
deleted_files = files_info[KEEP_COUNT:]

for file_info in deleted_files:
    os.remove(file_info[0])

print("\n--- Summary ---")
print(f"Kept {len(kept_files)} largest/highest quality images.")
print(f"Deleted {len(deleted_files)} excess images.")
print("The remaining 30 images should be the highest quality ones available.")
