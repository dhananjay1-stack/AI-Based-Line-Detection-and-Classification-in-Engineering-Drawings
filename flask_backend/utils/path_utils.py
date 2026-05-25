import os
import uuid
from pathlib import Path
from typing import Dict


def generate_job_id() -> str:
    return str(uuid.uuid4())


def get_job_input_dir(base_dir: str, job_id: str) -> str:
    path = os.path.join(base_dir, job_id, 'input')
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_job_output_dir(base_dir: str, job_id: str) -> str:
    path = os.path.join(base_dir, job_id)
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def get_output_subdirs(output_dir: str, mode: str) -> Dict[str, str]:
    subdirs = {}

    subdirs['input'] = os.path.join(output_dir, 'input')
    subdirs['debug_overlays'] = os.path.join(output_dir, 'debug_overlays')

    if mode in ['detection', 'both']:
        subdirs['binary_masks'] = os.path.join(output_dir, 'binary_masks')
        subdirs['masks'] = os.path.join(output_dir, 'masks')
        subdirs['vectors'] = os.path.join(output_dir, 'vectors')

    if mode in ['filter', 'both']:
        subdirs['filtered_images'] = os.path.join(output_dir, 'filtered_images')

    for path in subdirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    return subdirs


def get_relative_path(full_path: str, base_dir: str) -> str:
    return os.path.relpath(full_path, base_dir)


def get_api_file_path(job_id: str, relative_path: str) -> str:
    return f'/api/files/{job_id}/{relative_path}'


def get_download_path(job_id: str, file_type: str, filename: str = None) -> str:
    path = f'/api/download/{job_id}/{file_type}'
    if filename:
        path += f'?filename={filename}'
    return path


def get_basename(file_path: str) -> str:
    return Path(file_path).stem


def get_extension(file_path: str) -> str:
    return Path(file_path).suffix
