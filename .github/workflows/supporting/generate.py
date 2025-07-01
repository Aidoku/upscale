import os
import json
import shutil

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..')) # repo root
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
PUBLIC_DIR = os.path.join(ROOT_DIR, 'public')
PUBLIC_MODELS_DIR = os.path.join(PUBLIC_DIR, 'models')
OUTPUT_PATH = os.path.join(PUBLIC_DIR, 'models.json')

def get_path_size(path):
    if os.path.isfile(path):
        return os.path.getsize(path)
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total

# create public/models dir
os.makedirs(PUBLIC_MODELS_DIR, exist_ok=True)

models = []

for fname in os.listdir(MODELS_DIR):
    if fname.endswith('.json'):
        base = fname[:-5]
        mlmodel_path = os.path.join(MODELS_DIR, base + '.mlmodel')
        mlpackage_path = os.path.join(MODELS_DIR, base + '.mlpackage')
        model_file = None
        is_package = False
        if os.path.isfile(mlmodel_path):
            model_file = mlmodel_path
        elif os.path.isdir(mlpackage_path):
            model_file = mlpackage_path
            is_package = True
        else:
            continue
        # copy or zip model file to public/models
        if is_package:
            dest_model_file = os.path.join(PUBLIC_MODELS_DIR, base + '.mlpackage.zip')
            # Remove existing zip if present
            if os.path.exists(dest_model_file):
                os.remove(dest_model_file)
            shutil.make_archive(dest_model_file[:-4], 'zip', model_file)
            size = os.path.getsize(dest_model_file)
            file_field = f"models/{base}.mlpackage.zip"
        else:
            dest_model_file = os.path.join(PUBLIC_MODELS_DIR, os.path.basename(model_file))
            shutil.copy2(model_file, dest_model_file)
            size = os.path.getsize(model_file)
            file_field = f"models/{os.path.basename(model_file)}"
        # read json metadata
        with open(os.path.join(MODELS_DIR, fname), 'r') as f:
            meta = json.load(f)
        size = get_path_size(model_file)
        models.append({
            "name": meta.get("name", base),
            "info": meta.get("info"),
            "tags": meta.get("tags", []),
            "type": meta.get("type", "multiarray"),
            "config": meta.get("config", {}),
            "file": f"models/{os.path.basename(model_file)}",
            "size": size
        })

with open(OUTPUT_PATH, 'w') as f:
    f.write(json.dumps({"models": models}, separators=(',', ':')) + '\n')

# create CNAME for github pages
CNAME_PATH = os.path.join(PUBLIC_DIR, 'CNAME')
CNAME_DOMAIN = "upscale.aidoku.app"
with open(CNAME_PATH, 'w') as f:
    f.write(CNAME_DOMAIN + '\n')

# create index.html that redirects to github repo
INDEX_PATH = os.path.join(PUBLIC_DIR, 'index.html')
INDEX_HTML = """<!DOCTYPE html>
<html>
  <head>
    <meta http-equiv="refresh" content="0; url=https://github.com/Aidoku/upscale">
    <title>Redirecting...</title>
  </head>
  <body>
    <p>Redirecting to <a href="https://github.com/Aidoku/upscale">https://github.com/Aidoku/upscale</a></p>
  </body>
</html>
"""
with open(INDEX_PATH, 'w') as f:
    f.write(INDEX_HTML)
