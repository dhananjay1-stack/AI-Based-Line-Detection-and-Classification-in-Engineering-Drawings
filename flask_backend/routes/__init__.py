import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from routes.upload import create_upload_routes
from routes.process import create_process_routes
from routes.result import create_result_routes
from routes.download import create_download_routes

__all__ = [
    'create_upload_routes',
    'create_process_routes',
    'create_result_routes',
    'create_download_routes'
]
