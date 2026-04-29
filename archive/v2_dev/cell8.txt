import os, zipfile, shutil

OUTPUT_DIR = "/kaggle/working"
ZIP_PATH   = os.path.join(OUTPUT_DIR, "batik_v2_outputs.zip")

# Extensions worth keeping
KEEP_EXTS = {".keras", ".tflite", ".txt", ".csv", ".png", ".jpg"}

print(f"Zipping output files to: {ZIP_PATH}\n")

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if os.path.splitext(fname)[1].lower() not in KEEP_EXTS:
            continue
        if fname == os.path.basename(ZIP_PATH):
            continue
        zf.write(fpath, arcname=fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"  Added: {fname:55s} ({size_mb:.2f} MB)")

zip_size_mb = os.path.getsize(ZIP_PATH) / (1024 * 1024)
print(f"\nZip created: {ZIP_PATH}  ({zip_size_mb:.2f} MB)")
print("Download it from the Kaggle Output panel on the right.")
