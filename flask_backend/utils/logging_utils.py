import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger.handlers = []

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


app_logger = setup_logger('flask_backend')


def log_job_event(
    job_id: str,
    event: str,
    details: dict = None,
    level: str = 'info'
) -> None:
    message = f"[Job: {job_id}] {event}"
    if details:
        message += f" - {details}"

    log_func = getattr(app_logger, level, app_logger.info)
    log_func(message)


def log_processing_time(
    job_id: str,
    stage: str,
    duration: float,
    filename: Optional[str] = None
) -> None:
    if filename:
        app_logger.info(
            f"[Job: {job_id}] {stage} - {filename}: {duration:.2f}s"
        )
    else:
        app_logger.info(
            f"[Job: {job_id}] {stage}: {duration:.2f}s"
        )


def log_upload(job_id: str, file_count: int, filenames: list) -> None:
    app_logger.info(
        f"[Job: {job_id}] Upload: {file_count} file(s) - {filenames}"
    )


def log_processing_start(job_id: str, mode: str, file_count: int) -> None:
    app_logger.info(
        f"[Job: {job_id}] Processing started - Mode: {mode}, Files: {file_count}"
    )


def log_processing_complete(job_id: str, total_time: float) -> None:
    app_logger.info(
        f"[Job: {job_id}] Processing complete - Total time: {total_time:.2f}s"
    )


def log_processing_failed(job_id: str, error: str) -> None:
    app_logger.error(
        f"[Job: {job_id}] Processing failed - Error: {error}"
    )


def log_inference(
    job_id: str,
    model_type: str,
    filename: str,
    duration: float
) -> None:
    app_logger.info(
        f"[Job: {job_id}] {model_type} inference - {filename}: {duration:.2f}s"
    )


def log_output_saved(job_id: str, output_type: str, path: str) -> None:
    app_logger.debug(
        f"[Job: {job_id}] Saved {output_type}: {path}"
    )
