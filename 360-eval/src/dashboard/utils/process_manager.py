"""
Process Management for Background Benchmark Processes

Handles:
- Lock file creation and tracking
- Cleanup of stale/orphaned processes from previous runs
- Process group management for clean termination
- Safe process killing with validation
"""

import os
import json
import logging
import psutil
import signal
import uuid
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Lock file directory
LOCK_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOCK_DIR.mkdir(exist_ok=True)


# ----------------------------------------
# Lock File Management
# ----------------------------------------

def create_lock_file(
    eval_id: str,
    pid: int,
    command: List[str],
    session_id: Optional[str] = None
) -> Path:
    """
    Create a lock file for a running benchmark process.

    Args:
        eval_id: Evaluation ID
        pid: Process ID
        command: Command line arguments
        session_id: Unique session identifier

    Returns:
        Path to lock file
    """
    if session_id is None:
        session_id = str(uuid.uuid4())

    lock_file = LOCK_DIR / f"benchmark_{eval_id}.lock"

    lock_data = {
        "pid": pid,
        "session_id": session_id,
        "eval_id": eval_id,
        "started_at": datetime.now().isoformat(),
        "command": " ".join(command)
    }

    try:
        with open(lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)
        logger.info(f"Created lock file for eval {eval_id}: {lock_file}")
        return lock_file
    except Exception as e:
        logger.error(f"Failed to create lock file: {e}")
        return None


def remove_lock_file(eval_id: str) -> bool:
    """
    Remove lock file for completed/cancelled process.

    Args:
        eval_id: Evaluation ID

    Returns:
        True if removed successfully
    """
    lock_file = LOCK_DIR / f"benchmark_{eval_id}.lock"

    try:
        if lock_file.exists():
            lock_file.unlink()
            logger.info(f"Removed lock file for eval {eval_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to remove lock file: {e}")
        return False


def read_lock_file(lock_file: Path) -> Optional[Dict]:
    """
    Read lock file data.

    Args:
        lock_file: Path to lock file

    Returns:
        Lock data dict or None if invalid
    """
    try:
        with open(lock_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read lock file {lock_file}: {e}")
        return None


def get_all_lock_files() -> List[Path]:
    """
    Get all lock files in the logs directory.

    Returns:
        List of lock file paths
    """
    try:
        return list(LOCK_DIR.glob("benchmark_*.lock"))
    except Exception as e:
        logger.error(f"Failed to list lock files: {e}")
        return []


# ----------------------------------------
# Process Validation
# ----------------------------------------

def is_process_alive(pid: int) -> bool:
    """
    Check if a process with given PID is alive.

    Args:
        pid: Process ID

    Returns:
        True if process exists
    """
    try:
        return psutil.pid_exists(pid)
    except Exception as e:
        logger.warning(f"Failed to check if PID {pid} exists: {e}")
        return False


def is_our_process(pid: int, signature: str) -> bool:
    """
    Verify that a process belongs to our application.

    Args:
        pid: Process ID
        signature: Expected string in command line (e.g., eval_id)

    Returns:
        True if process matches our signature
    """
    try:
        process = psutil.Process(pid)
        cmdline = " ".join(process.cmdline())
        return signature in cmdline
    except (psutil.NoSuchProcess, psutil.AccessDenied, Exception) as e:
        logger.warning(f"Failed to verify process {pid}: {e}")
        return False


# ----------------------------------------
# Process Termination
# ----------------------------------------

def kill_process(pid: int, force: bool = False) -> bool:
    """
    Kill a process by PID.

    Args:
        pid: Process ID
        force: Use SIGKILL instead of SIGTERM

    Returns:
        True if killed successfully
    """
    try:
        process = psutil.Process(pid)

        # Try graceful termination first
        if not force:
            logger.info(f"Sending SIGTERM to process {pid}")
            process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.info(f"Process {pid} terminated gracefully")
                return True
            except psutil.TimeoutExpired:
                logger.warning(f"Process {pid} did not terminate gracefully, forcing...")
                force = True

        # Force kill if needed
        if force:
            logger.info(f"Sending SIGKILL to process {pid}")
            process.kill()
            process.wait(timeout=3)
            logger.info(f"Process {pid} killed forcefully")
            return True

    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} already dead")
        return True
    except Exception as e:
        logger.error(f"Failed to kill process {pid}: {e}")
        return False


def kill_process_group(pid: int) -> bool:
    """
    Kill a process and all its children.

    Args:
        pid: Parent process ID

    Returns:
        True if killed successfully
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Kill children first
        for child in children:
            try:
                logger.info(f"Killing child process {child.pid}")
                kill_process(child.pid, force=False)
            except Exception as e:
                logger.warning(f"Failed to kill child {child.pid}: {e}")

        # Kill parent
        logger.info(f"Killing parent process {pid}")
        return kill_process(pid, force=False)

    except psutil.NoSuchProcess:
        logger.info(f"Process {pid} already dead")
        return True
    except Exception as e:
        logger.error(f"Failed to kill process group {pid}: {e}")
        return False


# ----------------------------------------
# Stale Process Cleanup
# ----------------------------------------

def cleanup_stale_processes() -> Dict[str, int]:
    """
    Clean up stale/orphaned processes from previous runs.

    Scans lock files and kills processes that are:
    - Still running but from a previous session
    - Verified to be our benchmark processes

    Returns:
        Dict with cleanup statistics
    """
    stats = {
        "lock_files_found": 0,
        "processes_killed": 0,
        "stale_locks_removed": 0,
        "errors": 0
    }

    logger.info("Starting stale process cleanup...")

    lock_files = get_all_lock_files()
    stats["lock_files_found"] = len(lock_files)

    if not lock_files:
        logger.info("No lock files found, nothing to clean up")
        return stats

    for lock_file in lock_files:
        try:
            lock_data = read_lock_file(lock_file)

            if not lock_data:
                # Invalid lock file, remove it
                logger.warning(f"Removing invalid lock file: {lock_file}")
                lock_file.unlink()
                stats["stale_locks_removed"] += 1
                continue

            pid = lock_data.get("pid")
            eval_id = lock_data.get("eval_id")
            started_at = lock_data.get("started_at")

            logger.info(f"Checking lock file for eval {eval_id} (PID: {pid}, started: {started_at})")

            # Check if process is still alive
            if not is_process_alive(pid):
                logger.info(f"Process {pid} is dead, removing stale lock file")
                lock_file.unlink()
                stats["stale_locks_removed"] += 1
                continue

            # Verify it's our process
            if not is_our_process(pid, eval_id):
                logger.warning(f"Process {pid} exists but doesn't match eval {eval_id}, removing lock file")
                lock_file.unlink()
                stats["stale_locks_removed"] += 1
                continue

            # Process is alive and matches - kill it
            logger.warning(f"Found orphaned process {pid} for eval {eval_id}, killing...")

            if kill_process_group(pid):
                stats["processes_killed"] += 1
                lock_file.unlink()
                stats["stale_locks_removed"] += 1
                logger.info(f"Successfully cleaned up orphaned process {pid}")
            else:
                logger.error(f"Failed to kill orphaned process {pid}")
                stats["errors"] += 1

        except Exception as e:
            logger.error(f"Error processing lock file {lock_file}: {e}")
            stats["errors"] += 1

    logger.info(f"Stale process cleanup completed: {stats}")
    return stats


# ----------------------------------------
# Manual Process Control
# ----------------------------------------

def cancel_evaluation(eval_id: str) -> bool:
    """
    Cancel a running evaluation by eval_id.

    Args:
        eval_id: Evaluation ID to cancel

    Returns:
        True if cancelled successfully
    """
    lock_file = LOCK_DIR / f"benchmark_{eval_id}.lock"

    if not lock_file.exists():
        logger.warning(f"No lock file found for eval {eval_id}")
        return False

    lock_data = read_lock_file(lock_file)
    if not lock_data:
        logger.error(f"Invalid lock file for eval {eval_id}")
        return False

    pid = lock_data.get("pid")

    logger.info(f"Cancelling evaluation {eval_id} (PID: {pid})...")

    # Kill the process group
    success = kill_process_group(pid)

    # Remove lock file
    if success:
        remove_lock_file(eval_id)
        logger.info(f"Successfully cancelled evaluation {eval_id}")
    else:
        logger.error(f"Failed to cancel evaluation {eval_id}")

    return success


def get_running_evaluations() -> List[Dict]:
    """
    Get list of currently running evaluations.

    Returns:
        List of dicts with eval info
    """
    running = []

    for lock_file in get_all_lock_files():
        lock_data = read_lock_file(lock_file)

        if not lock_data:
            continue

        pid = lock_data.get("pid")

        # Check if process is actually running
        if is_process_alive(pid):
            running.append({
                "eval_id": lock_data.get("eval_id"),
                "pid": pid,
                "started_at": lock_data.get("started_at"),
                "command": lock_data.get("command")
            })

    return running
