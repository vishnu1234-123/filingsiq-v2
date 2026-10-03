DROP_EXACT = {
    "docling", "docling-core", "docling-ibm-models", "docling-parse", "docling-slim",
    "onnxruntime", "opencv-python", "rapidocr", "rtree", "shapely",
    "pyclipper", "latex2mathml", "pylatexenc", "semchunk", "mail-parser",
}
DROP_PREFIX_MATCH_LINE = "-e git+"

with open("requirements.txt") as f:
    lines=[l.rstrip("\n") for l in f if l.strip()]

kept, dropped = [], []
for line in lines:
    if line.startswith(DROP_PREFIX_MATCH_LINE):
        dropped.append(line)
        continue
    pkg_name = line.split("==")[0].strip()
    if pkg_name in DROP_EXACT:
        dropped.append(line)
        continue
    kept.append(line)
 
with open("requirements.txt", "w") as f:
    f.write("\n".join(kept) + "\n")
 
print(f"Kept {len(kept)} packages, dropped {len(dropped)}:")
for d in dropped:
    print(f"  - {d}")
 
