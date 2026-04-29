import os

target_dir = r"C:\Users\Benny Pepper\Downloads\Batik_dataset\merak_nigbing_downloaded"

print(f"Renaming files in: {target_dir}")

count = 1
for filename in os.listdir(target_dir):
    file_path = os.path.join(target_dir, filename)
    
    if os.path.isfile(file_path):
        ext = os.path.splitext(filename)[1].lower()
        new_filename = f"merak_ngibing_{count}{ext}"
        new_file_path = os.path.join(target_dir, new_filename)
        
        # Handle conflicts just in case
        while os.path.exists(new_file_path):
            count += 1
            new_filename = f"merak_ngibing_{count}{ext}"
            new_file_path = os.path.join(target_dir, new_filename)
            
        os.rename(file_path, new_file_path)
        print(f"Renamed: {filename} -> {new_filename}")
        count += 1

print("\nAll files renamed successfully!")
