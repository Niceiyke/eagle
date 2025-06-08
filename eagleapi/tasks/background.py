# eagle/tasks/background.py

import asyncio
import threading
import logging
import time
import uuid
import json
from datetime import datetime, timedelta
from queue import PriorityQueue, Empty, Queue
from typing import Callable, Any, Dict, Optional, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import signal
import sys

from eagleapi.tasks.model import TaskQueue as DBTask, TaskStatusEnum, TaskPriorityEnum
from eagleapi.db import db
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger("eagle.tasks")

def run_async_db(coro):
    """Run async DB coroutine from a thread."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # Running inside an event loop (shouldn't happen in thread)
        fut = asyncio.run_coroutine_threadsafe(coro, loop)
        return fut.result()
    else:
        return asyncio.run(coro)


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class TaskPriority(Enum):
    LOW = 10
    NORMAL = 5
    HIGH = 2
    CRITICAL = 1

@dataclass
class TaskConfig:
    """Configuration for task execution"""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = 300  # 5 minutes
    priority: TaskPriority = TaskPriority.NORMAL

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: Optional[str] = None
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    retry_count: int = 0
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'status': self.status.value,
            'result': self.result,
            'error': self.error,
            'created_at': self.created_at,
            'started_at': self.started_at,
            'completed_at': self.completed_at,
            'retry_count': self.retry_count,
            'execution_time': self.execution_time
        }

class TaskTimeoutError(Exception):
    """Raised when a task exceeds its timeout"""
    pass

class TaskCancellationError(Exception):
    """Raised when a task is cancelled"""
    pass

class BackgroundTaskQueue:
    """Enhanced background task queue with improved error handling and monitoring"""
    
    def __init__(self, 
                 num_workers: int = 4,
                 max_queue_size: int = 1000,
                 cleanup_interval: int = 3600,  # 1 hour
                 max_task_age: int = 86400):    # 24 hours
        
        self.queue = PriorityQueue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.cleanup_interval = cleanup_interval
        self.max_task_age = max_task_age
        
        # Task storage and tracking
        self.task_results: Dict[str, TaskResult] = {}
        self.cancelled_tasks: set = set()
        
        # Worker management
        self.workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Metrics
        self.metrics = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_cancelled': 0,
            'total_execution_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Start workers and cleanup
        self._start_workers()
        self._start_cleanup_thread()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGTERM/SIGINT"""
        def signal_handler(signum, frame):
            logger.info(f"[TaskQueue] Received signal {signum}, shutting down...")
            self.shutdown()
            sys.exit(0)
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker, 
                name=f"TaskWorker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"[TaskQueue] Worker {i + 1} started")

    def _start_cleanup_thread(self):
        """Start cleanup thread for old tasks"""
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_worker,
            name="TaskCleanup",
            daemon=True
        )
        self._cleanup_thread.start()
        logger.info("[TaskQueue] Cleanup thread started")

    def _worker(self):
        """Worker thread main loop"""
        while not self._shutdown_event.is_set():
            try:
                # Get task with timeout to allow periodic shutdown checks
                try:
                    priority, task_id, func, args, kwargs, config = self.queue.get(timeout=1.0)
                except Empty:
                    continue

                # Check if task was cancelled
                if task_id in self.cancelled_tasks:
                    self._update_task_status(task_id, TaskStatus.CANCELLED)
                    self.queue.task_done()
                    continue

                # Execute task
                self._execute_task(task_id, func, args, kwargs, config)
                self.queue.task_done()

            except Exception as e:
                logger.error(f"[TaskQueue] Worker error: {e}", exc_info=True)

    def _execute_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict, config: TaskConfig):
        """Execute a single task with retry logic and timeout"""
        task_result = self.task_results.get(task_id)
        if not task_result:
            logger.error(f"[TaskQueue] Task {task_id} not found in results")
            return

        start_time = time.time()
        task_result.started_at = start_time
        self._update_task_status(task_id, TaskStatus.RUNNING)

        for attempt in range(1, config.max_retries + 1):
            if task_id in self.cancelled_tasks:
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                return

            try:
                # Update retry status
                if attempt > 1:
                    self._update_task_status(task_id, TaskStatus.RETRYING)
                    task_result.retry_count = attempt - 1

                # Execute with timeout
                result = self._execute_with_timeout(func, args, kwargs, config.timeout)

                # Success
                execution_time = time.time() - start_time
                task_result.execution_time = execution_time
                task_result.completed_at = time.time()
                task_result.result = result

                # Update DB result
                async def update_db_result():
                    async with db.get_session() as session:
                        db_task = await session.get(DBTask, task_id)
                        if db_task:
                            db_task.result = result
                            db_task.status = TaskStatusEnum.SUCCESS
                            db_task.finished_at = datetime.utcnow()
                            await session.commit()
                run_async_db(update_db_result())

                self._update_task_status(task_id, TaskStatus.SUCCESS)

                # Update metrics
                with self._lock:
                    self.metrics['tasks_completed'] += 1
                    self.metrics['total_execution_time'] += execution_time
                
                logger.info(f"[TaskQueue] Task {task_id} completed successfully in {execution_time:.2f}s")
                return

            except TaskTimeoutError as e:
                error_msg = f"Task timed out after {config.timeout}s"
                logger.warning(f"[TaskQueue] Task {task_id} attempt {attempt}/{config.max_retries}: {error_msg}")
                
            except TaskCancellationError:
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                return
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[TaskQueue] Task {task_id} attempt {attempt}/{config.max_retries}: {error_msg}")
                
                if attempt == config.max_retries:
                    # Final failure
                    task_result.error = error_msg
                    task_result.completed_at = time.time()
                    task_result.execution_time = time.time() - start_time
                    task_result.retry_count = attempt - 1
                    
                    # Update DB error
                    async def update_db_error():
                        async with db.get_session() as session:
                            db_task = await session.get(DBTask, task_id)
                            if db_task:
                                db_task.error = error_msg
                                db_task.status = TaskStatusEnum.FAILED
                                db_task.finished_at = datetime.utcnow()
                                await session.commit()
                    run_async_db(update_db_error())

                    self._update_task_status(task_id, TaskStatus.FAILED)

                    with self._lock:
                        self.metrics['tasks_failed'] += 1
                    
                    logger.error(f"[TaskQueue] Task {task_id} failed permanently: {error_msg}")
                    return
                else:
                    # Wait before retry
                    time.sleep(config.retry_delay * attempt)  # Exponential backoff

    def _execute_with_timeout(self, func: Callable, args: tuple, kwargs: dict, timeout: Optional[float]) -> Any:
        """Execute function with timeout"""
        if timeout is None:
            return func(*args, **kwargs)

        result_queue = Queue()
        exception_queue = Queue()

        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)

        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred - we can't actually kill the thread, but we can abandon it
            raise TaskTimeoutError(f"Task exceeded timeout of {timeout} seconds")

        if not exception_queue.empty():
            raise exception_queue.get()

        if not result_queue.empty():
            return result_queue.get()

        raise Exception("Task completed but no result was produced")

    def _update_task_status(self, task_id: str, status: TaskStatus):
        """Update task status thread-safely and persist to DB."""
        with self._lock:
            if task_id in self.task_results:
                self.task_results[task_id].status = status
        # Update in DB
        async def update_db_status():
            async with db.get_session() as session:
                db_task = await session.get(DBTask, task_id)
                if db_task:
                    db_task.status = TaskStatusEnum(status.value)
                    if status == TaskStatus.RUNNING:
                        db_task.started_at = datetime.utcnow()
                    elif status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                        db_task.finished_at = datetime.utcnow()
                    await session.commit()
        run_async_db(update_db_status())

    def _cleanup_worker(self):
        """Cleanup old completed tasks periodically"""
        while not self._shutdown_event.is_set():
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_tasks()
            except Exception as e:
                logger.error(f"[TaskQueue] Cleanup error: {e}", exc_info=True)

    def _cleanup_old_tasks(self):
        """Remove old completed tasks to prevent memory leaks"""
        cutoff_time = time.time() - self.max_task_age
        
        with self._lock:
            to_remove = []
            for task_id, task_result in self.task_results.items():
                if (task_result.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                    task_result.completed_at and task_result.completed_at < cutoff_time):
                    to_remove.append(task_id)

            for task_id in to_remove:
                del self.task_results[task_id]
                self.cancelled_tasks.discard(task_id)

            if to_remove:
                logger.info(f"[TaskQueue] Cleaned up {len(to_remove)} old tasks")

    def submit(self, 
               func: Callable,
               *args: Any,
               config: Optional[TaskConfig] = None,
               **kwargs: Any) -> str:
        """Submit a task for background execution"""
        
        if not callable(func):
            raise ValueError("Function must be callable")

        if config is None:
            config = TaskConfig()

        # Check queue capacity
        if self.queue.qsize() >= self.max_queue_size:
            raise RuntimeError(f"Task queue is full (max: {self.max_queue_size})")

        task_id = str(uuid.uuid4())

        # Persist to DB
        async def create_db_task():
            async with db.get_session() as session:
                db_task = DBTask(
                    id=task_id,
                    status=TaskStatusEnum.PENDING,
                    priority=TaskPriorityEnum(config.priority.value),
                    func_name=func.__name__,
                    args=args,
                    kwargs=kwargs,
                    result=None,
                    error=None,
                    started_at=None,
                    finished_at=None
                )
                session.add(db_task)
                await session.commit()
        run_async_db(create_db_task())

        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.PENDING,
            created_at=time.time()
        )

        with self._lock:
            self.task_results[task_id] = task_result
            self.metrics['tasks_submitted'] += 1

        # Submit to queue
        self.queue.put((config.priority.value, task_id, func, args, kwargs, config))
        
        logger.info(f"[TaskQueue] Task submitted: {task_id} ({func.__name__}) priority={config.priority.name}")
        return task_id

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            if task_id not in self.task_results:
                return False

            task_result = self.task_results[task_id]
            
            if task_result.status in [TaskStatus.SUCCESS, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return False  # Already completed

            self.cancelled_tasks.add(task_id)
            
            if task_result.status == TaskStatus.PENDING:
                # If pending, update status immediately
                self._update_task_status(task_id, TaskStatus.CANCELLED)
                self.metrics['tasks_cancelled'] += 1

            logger.info(f"[TaskQueue] Task {task_id} marked for cancellation")
            return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result"""
        with self._lock:
            task_result = self.task_results.get(task_id)
            return task_result.to_dict() if task_result else None

    def get_all_tasks(self, status_filter: Optional[TaskStatus] = None) -> Dict[str, Dict[str, Any]]:
        """Get all tasks, optionally filtered by status"""
        with self._lock:
            if status_filter:
                return {
                    task_id: task_result.to_dict()
                    for task_id, task_result in self.task_results.items()
                    if task_result.status == status_filter
                }
            return {task_id: task_result.to_dict() for task_id, task_result in self.task_results.items()}

    def get_queue_info(self) -> Dict[str, Any]:
        """Get queue information and metrics"""
        with self._lock:
            status_counts = {}
            for task_result in self.task_results.values():
                status = task_result.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            avg_execution_time = (
                self.metrics['total_execution_time'] / max(self.metrics['tasks_completed'], 1)
            )

            return {
                'queue_size': self.queue.qsize(),
                'total_tasks': len(self.task_results),
                'status_breakdown': status_counts,
                'active_workers': len([w for w in self.workers if w.is_alive()]),
                'metrics': {
                    **self.metrics,
                    'average_execution_time': round(avg_execution_time, 2)
                },
                'configuration': {
                    'num_workers': self.num_workers,
                    'max_queue_size': self.max_queue_size,
                    'cleanup_interval': self.cleanup_interval,
                    'max_task_age': self.max_task_age
                }
            }

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while True:
            task_status = self.get_task_status(task_id)
            if not task_status:
                return None

            if task_status['status'] in ['success', 'failed', 'cancelled']:
                return task_status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            time.sleep(0.1)  # Poll every 100ms

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Gracefully shutdown the task queue"""
        logger.info("[TaskQueue] Initiating shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.workers:
            remaining_time = max(0, timeout - (time.time() - start_time))
            worker.join(timeout=remaining_time)
            
            if worker.is_alive():
                logger.warning(f"[TaskQueue] Worker {worker.name} did not shutdown gracefully")

        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self._cleanup_thread.join(timeout=remaining_time)

        # Final cleanup
        self._cleanup_old_tasks()
        
        shutdown_time = time.time() - start_time
        logger.info(f"[TaskQueue] Shutdown completed in {shutdown_time:.2f}s")
        
        return shutdown_time <= timeout

    @contextmanager
    def task_context(self, func: Callable, *args, config: Optional[TaskConfig] = None, **kwargs):
        """Context manager for task submission and waiting"""
        task_id = self.submit(func, *args, config=config, **kwargs)
        try:
            yield task_id
        finally:
            # Optionally cancel if still running
            pass

# Convenience functions for common task configurations
def create_high_priority_config(timeout: float = 60) -> TaskConfig:
    return TaskConfig(priority=TaskPriority.HIGH, timeout=timeout, max_retries=1)

def create_reliable_config(max_retries: int = 5) -> TaskConfig:
    return TaskConfig(max_retries=max_retries, retry_delay=2.0, timeout=600)

def create_quick_config() -> TaskConfig:
    return TaskConfig(timeout=30, max_retries=1, priority=TaskPriority.NORMAL)