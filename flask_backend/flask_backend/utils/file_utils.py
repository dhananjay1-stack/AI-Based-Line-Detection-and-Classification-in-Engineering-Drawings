import os
import shutil
import zipfile
from pathlib import Path
from typing import List, Set
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS: Set[str] = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}


def allowed_file(filename: str, extensions: Set[str] = None) -> bool:
    if extensions is None:
        extensions = ALLOWED_EXTENSIONS
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions


def secure_save_file(file, directory: str) -> str:
    filename = secure_filename(file.filename)
    file_path = os.path.join(directory, filename)
    file.save(file_path)
    return file_path


def extract_zip(zip_path: str, extract_dir: str) -> List[str]:
    extracted_images = []

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    os.remove(zip_path)

    for root, dirs, files in os.walk(extract_dir):
        for filename in files:
            if allowed_file(filename):
                extracted_images.append(os.path.join(root, filename))

    return extracted_images


def list_image_files(directory: str, extensions: List[str] = None) -> List[str]:
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']

    files = []
    for ext in extensions:
        files.extend(Path(directory).glob(f'*{ext}'))
        files.extend(Path(directory).glob(f'*{ext.upper()}'))

    return sorted([str(f) for f in files])


def ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def clean_directory(directory: str) -> None:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)


def get_file_size(file_path: str) -> int:
    return os.path.getsize(file_path)


def copy_file(src: str, dst: str) -> str:
    shutil.copy2(src, dst)
    return dst
