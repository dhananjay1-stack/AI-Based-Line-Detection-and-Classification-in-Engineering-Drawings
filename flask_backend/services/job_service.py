import os
import shutil
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any


class JobService:
    STATUS_UPLOADED = 'uploaded'
    STATUS_QUEUED = 'queued'
    STATUS_PROCESSING = 'processing'
    STATUS_COMPLETED = 'completed'
    STATUS_FAILED = 'failed'

    def __init__(self, upload_folder: str, output_folder: str):
        self.upload_folder = upload_folder
        self.output_folder = output_folder
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()

    def create_job(
        self,
        job_id: str,
        files: List[str],
        filenames: List[str]
    ) -> Dict[str, Any]:
        with self.lock:
            job = {
                'job_id': job_id,
                'status': self.STATUS_UPLOADED,
                'files': files,
                'file_count': len(files),
                'filenames': filenames,
                'mode': None,
                'created_at': datetime.now().isoformat(),
                'started_at': None,
                'completed_at': None,
                'progress': None,
                'results': None,
                'error': None,
                'warnings': []
            }
            self.jobs[job_id] = job
            return job.copy()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job by ID."""
        with self.lock:
            job = self.jobs.get(job_id)
            return job.copy() if job else None

    def job_exists(self, job_id: str) -> bool:
        """Check if job exists."""
        return job_id in self.jobs

    def update_status(
        self,
        job_id: str,
        status: str,
        error: str = None
    ) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['status'] = status
                if error:
                    self.jobs[job_id]['error'] = error
                if status == self.STATUS_PROCESSING:
                    self.jobs[job_id]['started_at'] = datetime.now().isoformat()
                elif status in [self.STATUS_COMPLETED, self.STATUS_FAILED]:
                    self.jobs[job_id]['completed_at'] = datetime.now().isoformat()

    def set_mode(self, job_id: str, mode: str) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['mode'] = mode

    def update_progress(
        self,
        job_id: str,
        current: int,
        total: int,
        filename: str
    ) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = {
                    'current': current,
                    'total': total,
                    'filename': filename,
                    'percentage': round(100 * current / total, 1) if total > 0 else 0
                }

    def set_results(self, job_id: str, results: List[Dict]) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['results'] = results

    def add_warning(self, job_id: str, warning: str) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]['warnings'].append(warning)

    def get_job_dir(self, job_id: str) -> str:
        return os.path.join(self.upload_folder, job_id)

    def get_output_dir(self, job_id: str) -> str:
        return os.path.join(self.output_folder, job_id)

    def create_job_directories(self, job_id: str, mode: str) -> Dict[str, str]:
        output_dir = self.get_output_dir(job_id)
        dirs = {
            'base': output_dir,
            'input': os.path.join(output_dir, 'input')
        }

        if mode in ['detection', 'both']:
            dirs['masks'] = os.path.join(output_dir, 'masks')
            dirs['binary_masks'] = os.path.join(output_dir, 'binary_masks')

        if mode in ['filter', 'both']:
            dirs['filtered_images'] = os.path.join(output_dir, 'filtered_images')

        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        return dirs

    def can_process(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
            return job['status'] in [self.STATUS_UPLOADED, self.STATUS_COMPLETED, self.STATUS_FAILED]

    def is_processing(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            return job and job['status'] == self.STATUS_PROCESSING

    def get_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self.lock:
            history = []
            for job_id, job in self.jobs.items():
                history.append({
                    'job_id': job_id,
                    'status': job['status'],
                    'mode': job.get('mode', 'unknown'),
                    'file_count': job.get('file_count', 0),
                    'created_at': job.get('created_at'),
                    'completed_at': job.get('completed_at')
                })

            history.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return history[:limit]

    def delete_job(self, job_id: str) -> bool:
        with self.lock:
            if job_id not in self.jobs:
                return False

            del self.jobs[job_id]
        job_dir = self.get_job_dir(job_id)
        output_dir = self.get_output_dir(job_id)

        if os.path.exists(job_dir):
            shutil.rmtree(job_dir, ignore_errors=True)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir, ignore_errors=True)

        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24) -> int:
        from datetime import timedelta

        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        jobs_to_delete = []

        with self.lock:
            for job_id, job in self.jobs.items():
                created_str = job.get('created_at', '')
                if created_str:
                    try:
                        created = datetime.fromisoformat(created_str)
                        if created < cutoff:
                            jobs_to_delete.append(job_id)
                    except ValueError:
                        pass

        for job_id in jobs_to_delete:
            self.delete_job(job_id)

        return len(jobs_to_delete)
