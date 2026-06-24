import os
import shutil

def download_custom_hook(cfg) -> str | None:
    """
    Download or copy a single custom hook file from the --hook-source argument.
    Returns: the absolute path to the local file, or None if it fails / is missing.
    """
    source = cfg.hook_source
    if not source:
        return None

    cache_dir = os.path.join(cfg.outputs_dir, "hooks_cache")
    os.makedirs(cache_dir, exist_ok=True)
    local_path = os.path.join(cache_dir, "custom_hook_override.mp4")

    # 1. Download if URL
    if source.startswith("http"):
        print(f"📥 Downloading single custom hook from: {source}")
        import gdown
        try:
            # Gdown download for a single file
            file_id = source.split('/d/')[1].split('/')[0] if '/d/' in source else source
            download_url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(download_url, local_path, quiet=False)
            if os.path.exists(local_path):
                print(f"   ✅ Custom hook successfully downloaded to {local_path}")
                return local_path
            else:
                # Fallback, maybe it's not a google drive ID but a direct .mp4 link
                import requests
                r = requests.get(source, stream=True)
                if r.status_code == 200:
                    with open(local_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)
                    print(f"   ✅ Custom hook successfully downloaded to {local_path}")
                    return local_path
                return None
        except Exception as e:
            print(f"⚠️ Failed to download custom hook: {e}")
            return None
    else:
        # It's a local path
        if not os.path.exists(source):
            print(f"⚠️ Local source hook not found: {source}")
            return None
        
        print(f"📥 Using local hook file: {source}")
        try:
            shutil.copy(source, local_path)
            return local_path
        except shutil.SameFileError:
            return source
        except Exception as e:
            print(f"⚠️ Failed to copy local hook file: {e}")
            return source

