# eagle/tasks/background.py

import asyncio
import threading
import logging
import queue
import time
import uuid
import json
from datetime import datetime, timedelta
from queue import PriorityQueue, Empty, Queue
from typing import Callable, Any, Dict, Optional, List, Union, Awaitable
from dataclasses import dataclass, asdict
from enum import Enum
from contextlib import contextmanager
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
import functools
import asyncio
import inspect
from sqlalchemy import select

from eagleapi.tasks.model import TaskQueue, TaskStatusEnum, TaskPriorityEnum
from eagleapi.db import db
from sqlalchemy.ext.asyncio import AsyncSession
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("eagle.tasks")

# Use enums from model.py
from eagleapi.tasks.model import TaskStatusEnum, TaskPriorityEnum

@dataclass
class TaskConfig:
    """Configuration for task execution"""
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = 300  # 5 minutes
    priority: TaskPriorityEnum = TaskPriorityEnum.NORMAL

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatusEnum
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
        'status': self.status.value if hasattr(self.status, 'value') else self.status,
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
    """Enhanced background task queue with non-blocking database operations"""
    
    def __init__(self, 
                 num_workers: int = 4,
                 max_queue_size: int = 1000,
                 cleanup_interval: int = 3600,  # 1 hour
                 max_task_age: int = 86400,     # 24 hours
                 enable_db_persistence: bool = True):
        
        self.queue = PriorityQueue(maxsize=max_queue_size)
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        self.cleanup_interval = cleanup_interval
        self.max_task_age = max_task_age
        self.enable_db_persistence = enable_db_persistence
        
        # Task storage and tracking
        self.task_results: Dict[str, TaskResult] = {}
        self.cancelled_tasks: set = set()
        
        # Worker management
        self.workers: List[threading.Thread] = []
        self._shutdown_event = threading.Event()
        self._cleanup_thread: Optional[threading.Thread] = None
        
        # Database operation queue (non-blocking)
        self._db_queue: Queue = Queue()
        self._db_worker_thread: Optional[threading.Thread] = None
        
        # Event loop for async operations
        self._async_loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_thread: Optional[threading.Thread] = None
        
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
        
        # Start all components
        self._start_async_loop()
        self._start_db_worker()
        self._start_workers()
        self._start_cleanup_thread()
        self._setup_signal_handlers()

    def _start_async_loop(self):
        """Start a dedicated event loop thread for async operations"""
        def run_loop():
            self._async_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._async_loop)
            try:
                self._async_loop.run_forever()
            except Exception as e:
                logger.error(f"[TaskQueue] Async loop error: {e}")
            finally:
                self._async_loop.close()
        
        self._loop_thread = threading.Thread(target=run_loop, name="AsyncLoop", daemon=True)
        self._loop_thread.start()
        
        # Wait for loop to be ready
        while self._async_loop is None:
            time.sleep(0.01)
        
        logger.info("[TaskQueue] Async event loop started")

    def db_worker(self):
        """Database worker that processes database operations asynchronously"""
        logger.info("[TaskQueue] Database worker started")
        
        while not self._shutdown_event.is_set():
            try:
                # Get operation from queue with timeout
                try:
                    operation = self._db_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Run the operation in the event loop
                try:
                    if self._async_loop and not self._async_loop.is_closed():
                        future = asyncio.run_coroutine_threadsafe(
                            self._run_db_operation(operation),
                            self._async_loop
                        )
                        # Wait for the operation to complete with a timeout
                        future.result(timeout=30.0)  # 30 seconds timeout for DB operations
                    else:
                        logger.error("[TaskQueue] No event loop available for DB operation")
                except asyncio.TimeoutError:
                    logger.error("[TaskQueue] DB operation timed out")
                except Exception as e:
                    logger.error(f"[TaskQueue] Error in DB operation: {e}", exc_info=True)
                finally:
                    self._db_queue.task_done()
            except Exception as e:
                logger.error(f"[TaskQueue] Unexpected error in DB worker: {e}", exc_info=True)
                time.sleep(1)  # Prevent tight loop on errors
        
        logger.info("[TaskQueue] Database worker shutting down")
    
    async def _update_db_status(self, task_id: str, status: TaskStatusEnum):
        """Update task status in database with robust error handling"""
        if not self.enable_db_persistence:
            return
            
        max_retries = 3
        base_delay = 0.05  # 50ms base delay
        
        for attempt in range(max_retries):
            try:
                async with db.session() as session:
                    db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                    
                    if not db_task:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"[TaskQueue] Task {task_id} not found in DB, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"[TaskQueue] Task {task_id} not found in DB after {max_retries} attempts")
                            return
                    
                    # Check if we should update (prevent backwards status updates)
                    if not self._should_update_status(db_task.status, status):
                        logger.debug(f"[TaskQueue] Skipping status update for {task_id}: {db_task.status} -> {status}")
                        return
                    
                    # Update the status
                    old_status = db_task.status
                    db_task.status = status
                    
                    if status == TaskStatusEnum.RUNNING and not db_task.started_at:
                        db_task.started_at = datetime.utcnow()
                    elif status in [TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]:
                        if not db_task.finished_at:
                            db_task.finished_at = datetime.utcnow()
                    
                    await session.commit()
                    logger.info(f"[TaskQueue] Updated DB status for {task_id}: {old_status} -> {status}")
                    return  # Success, exit retry loop
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[TaskQueue] Failed to update DB status for task {task_id} after {max_retries} attempts: {e}", exc_info=True)
                else:
                    logger.warning(f"[TaskQueue] DB status update failed for task {task_id}, attempt {attempt + 1}: {e}")
                    await asyncio.sleep(base_delay * (2 ** attempt))

    async def _update_db_result(self, task_id: str, result: Any, status: TaskStatusEnum):
        """Update task result in database with robust error handling"""
        if not self.enable_db_persistence:
            return
            
        max_retries = 3
        base_delay = 0.05
        
        for attempt in range(max_retries):
            try:
                async with db.session() as session:
                    db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                    
                    if not db_task:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"[TaskQueue] Task {task_id} not found for result update, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"[TaskQueue] Task {task_id} not found for result update after {max_retries} attempts")
                            return
                    
                    # Only update if status progression is valid
                    if not self._should_update_status(db_task.status, status):
                        logger.debug(f"[TaskQueue] Skipping result update for {task_id} due to status check")
                        return
                    
                    # Update result and status
                    db_task.result = result
                    db_task.status = status
                    if not db_task.finished_at:
                        db_task.finished_at = datetime.utcnow()
                    
                    await session.commit()
                    logger.info(f"[TaskQueue] Updated DB result for {task_id} with status {status}")
                    return
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[TaskQueue] Failed to update DB result for task {task_id} after {max_retries} attempts: {e}", exc_info=True)
                else:
                    logger.warning(f"[TaskQueue] DB result update failed for task {task_id}, attempt {attempt + 1}: {e}")
                    await asyncio.sleep(base_delay * (2 ** attempt))

    async def _update_db_error(self, task_id: str, error: str, status: TaskStatusEnum):
        """Update task error in database with robust error handling"""
        if not self.enable_db_persistence:
            return
            
        max_retries = 3
        base_delay = 0.05
        
        for attempt in range(max_retries):
            try:
                async with db.session() as session:
                    db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                    
                    if not db_task:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)
                            logger.warning(f"[TaskQueue] Task {task_id} not found for error update, retrying in {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error(f"[TaskQueue] Task {task_id} not found for error update after {max_retries} attempts")
                            return
                    
                    # Only update if status progression is valid
                    if not self._should_update_status(db_task.status, status):
                        logger.debug(f"[TaskQueue] Skipping error update for {task_id} due to status check")
                        return
                    
                    # Update error and status
                    db_task.error = error
                    db_task.status = status
                    if not db_task.finished_at:
                        db_task.finished_at = datetime.utcnow()
                    
                    await session.commit()
                    logger.info(f"[TaskQueue] Updated DB error for {task_id} with status {status}")
                    return
                    
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"[TaskQueue] Failed to update DB error for task {task_id} after {max_retries} attempts: {e}", exc_info=True)
                else:
                    logger.warning(f"[TaskQueue] DB error update failed for task {task_id}, attempt {attempt + 1}: {e}")
                    await asyncio.sleep(base_delay * (2 ** attempt))

    async def _run_db_operation(self, operation: Callable[[], Awaitable[None]]):
        """Run a database operation with error handling"""
        try:
            await operation()
        except Exception as e:
            logger.error(f"[TaskQueue] Database operation failed: {e}", exc_info=True)
            raise
    
    def _start_db_worker(self):
        """Start database worker thread for non-blocking DB operations"""
        if not self.enable_db_persistence:
            logger.info("[TaskQueue] Database persistence disabled, not starting DB worker")
            return
            
        self._db_worker_thread = threading.Thread(
            target=self.db_worker,
            name="DBWorker",
            daemon=True
        )
        self._db_worker_thread.start()
        logger.info("[TaskQueue] Database worker started")

    def _queue_db_operation(self, operation: Callable[[], Any]):
        """Queue a database operation for non-blocking execution"""
        if not self.enable_db_persistence:
            return
            
        if not self._db_worker_thread or not self._db_worker_thread.is_alive():
            logger.warning("[TaskQueue] Database worker not running, operation not queued")
            return
            
        try:
            self._db_queue.put(operation)
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to queue DB operation: {e}", exc_info=True)

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
        logger.info(f"[TaskQueue] Starting {self.num_workers} worker threads")
        for i in range(self.num_workers):
            worker = threading.Thread(
                target=self._worker, 
                name=f"Worker-{i+1}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
            logger.info(f"[TaskQueue] Started worker thread: {worker.name} (ID: {worker.ident})")
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
        thread_name = threading.current_thread().name
        logger.info(f"[TaskQueue] Worker {thread_name} started and ready for tasks")
        
        while not self._shutdown_event.is_set():
            try:
                # Get task with timeout to allow shutdown check
                try:
                    logger.debug(f"[TaskQueue] {thread_name} waiting for task...")
                    priority, task_id, func, args, kwargs, config = self.queue.get(timeout=1.0)
                    logger.info(f"[TaskQueue] {thread_name} picked up task {task_id} ({func.__name__}) with priority {priority}")
                except queue.Empty:
                    logger.debug(f"[TaskQueue] {thread_name} no tasks in queue, waiting...")
                    continue

                # Check if task was cancelled
                if task_id in self.cancelled_tasks:
                    self._queue_db_operation(functools.partial(
                        self._update_db_status, task_id, TaskStatusEnum.CANCELLED
                    ))
                    self.queue.task_done()
                    continue

                # Execute task
                self._execute_task(task_id, func, args, kwargs, config)
                self.queue.task_done()

            except Exception as e:
                logger.error(f"[TaskQueue] Worker error: {e}", exc_info=True)

    def _execute_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict, config: TaskConfig):
        """Execute a single task with retry logic and timeout"""
        thread_name = threading.current_thread().name
        logger.info(f"[TaskQueue] {thread_name} executing task {task_id} ({func.__name__})")
        
        # Use lock when accessing task_results to prevent race conditions
        with self._lock:
            task_result = self.task_results.get(task_id)
            if not task_result:
                logger.error(f"[TaskQueue] Task {task_id} not found in results")
                return
                
            # Update task status to RUNNING
            task_result.status = TaskStatusEnum.RUNNING
            task_result.started_at = time.time()
            
            # Queue the database status update
            self._queue_db_operation(functools.partial(
                self._update_db_status, task_id, TaskStatusEnum.RUNNING
            ))

        start_time = time.time()

        for attempt in range(1, config.max_retries + 1):
            if task_id in self.cancelled_tasks:
                self._queue_db_operation(functools.partial(
                    self._update_db_status, task_id, TaskStatusEnum.CANCELLED
                ))
                return

            try:
                # Update retry status
                if attempt > 1:
                    self._queue_db_operation(functools.partial(
                        self._update_db_status, task_id, TaskStatusEnum.RETRYING
                    ))
                    task_result.retry_count = attempt - 1

                # Execute with timeout
                result = self._execute_with_timeout(func, args, kwargs, config.timeout)

                # Success - update task result under lock
                execution_time = time.time() - start_time
                with self._lock:
                    task_result.status = TaskStatusEnum.SUCCESS
                    task_result.result = result
                    task_result.completed_at = time.time()
                    task_result.execution_time = execution_time
                    
                    # Update metrics
                    self.metrics['tasks_completed'] += 1
                    self.metrics['total_execution_time'] += execution_time
                
                # Queue database updates
                self._queue_db_operation(functools.partial(
                    self._update_db_result, task_id, result, TaskStatusEnum.SUCCESS
                ))
                self._queue_db_operation(functools.partial(
                    self._update_db_status, task_id, TaskStatusEnum.SUCCESS
                ))
                
                logger.info(f"[TaskQueue] Task {task_id} completed successfully in {execution_time:.2f}s")
                return

            except TaskTimeoutError as e:
                error_msg = f"Task timed out after {config.timeout}s"
                logger.warning(f"[TaskQueue] Task {task_id} attempt {attempt}/{config.max_retries}: {error_msg}")
                
            except TaskCancellationError:
                self._queue_db_operation(functools.partial(
                    self._update_db_status, task_id, TaskStatusEnum.CANCELLED
                ))
                return
                
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"[TaskQueue] Task {task_id} attempt {attempt}/{config.max_retries}: {error_msg}")
                
                if attempt == config.max_retries:
                    # Final failure - update task result under lock
                    execution_time = time.time() - start_time
                    with self._lock:
                        task_result.status = TaskStatusEnum.FAILED
                        task_result.error = error_msg
                        task_result.completed_at = time.time()
                        task_result.execution_time = execution_time
                        task_result.retry_count = attempt - 1
                    
                    # Queue database updates
                    self._queue_db_operation(functools.partial(
                        self._update_db_error, task_id, error_msg, TaskStatusEnum.FAILED
                    ))
                    self._queue_db_operation(functools.partial(
                        self._update_db_status, task_id, TaskStatusEnum.FAILED
                    ))

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
            if inspect.iscoroutinefunction(func):
                return asyncio.run(func(*args, **kwargs))
            result = func(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return asyncio.run(result)
            return result

        result_queue = Queue()
        exception_queue = Queue()

        def target():
            try:
                if inspect.iscoroutinefunction(func):
                    res = asyncio.run(func(*args, **kwargs))
                else:
                    res = func(*args, **kwargs)
                    if asyncio.iscoroutine(res):
                        res = asyncio.run(res)
                result_queue.put(res)
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

    async def _create_db_task(self, task_id: str, func: Callable, args: tuple, kwargs: dict, config: TaskConfig):
        """Create task in database"""
        logger.info(f"[TaskQueue] Creating DB task {task_id}")
        try:
            # Convert args to list and ensure it's JSON-serializable
            import json
            args_list = list(args)
            # Ensure kwargs is a JSON-serializable dict
            kwargs_dict = dict(kwargs)
            
            # Test JSON serialization
            json.dumps(args_list)
            json.dumps(kwargs_dict)
            
            from .model import TaskQueue
            async with db.session() as session:
                # Check if task already exists
                existing_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                if existing_task:
                    logger.warning(f"[TaskQueue] Task {task_id} already exists in DB")
                    return
                    
                db_task = TaskQueue(
                    task_id=task_id,
                    status=TaskStatusEnum.PENDING,
                    priority=config.priority,
                    func_name=func.__name__,
                    args=args,  
                    kwargs=kwargs,  
                    result=None,
                    error=None,
                    started_at=None,
                    finished_at=None,
                    created_at=datetime.utcnow()
                )
                logger.info(f"[TaskQueue] Creating DB task {task_id} with status PENDING")
                session.add(db_task)
                await session.commit()
                logger.info(f"[TaskQueue] Successfully created DB task {task_id}")
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to create DB task {task_id}: {e}", exc_info=True)
            raise  # Re-raise to ensure the error is caught by the caller
    
    async def _update_db_status(self, task_id: str, status: TaskStatusEnum):
        """Update task status in database"""
        try:
            async with db.session() as session:
                db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                if db_task:
                    db_task.status = status
                    if status == TaskStatusEnum.RUNNING:
                        db_task.started_at = datetime.utcnow()
                    elif status in [TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]:
                        db_task.finished_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to update DB status for task {task_id}: {e}")

    async def _update_db_result(self, task_id: str, result: Any, status: TaskStatusEnum):
        """Update task result in database"""
        try:
            async with db.session() as session:
                db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                if db_task:
                    db_task.result = result
                    db_task.status = status
                    db_task.finished_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to update DB result for task {task_id}: {e}")

    async def _update_db_error(self, task_id: str, error: str, status: TaskStatusEnum):
        """Update task error in database"""
        try:
            async with db.session() as session:
                db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                if db_task:
                    db_task.error = error
                    db_task.status = status
                    db_task.finished_at = datetime.utcnow()
                    await session.commit()
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to update DB error for task {task_id}: {e}")

    def _cleanup_worker(self):
        """Cleanup old completed tasks periodically"""
        while not self._shutdown_event.is_set():
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_old_tasks()
            except Exception as e:
                logger.error(f"[TaskQueue] Cleanup error: {e}", exc_info=True)

    async def _cleanup_failed_task(self, task_id: str):
        """Clean up a task that failed to be queued"""
        try:
            async with db.session() as session:
                db_task = await session.scalar(select(TaskQueue).filter_by(task_id=task_id))
                if db_task:
                    await session.delete(db_task)
                    await session.commit()
                    logger.info(f"[TaskQueue] Cleaned up failed task {task_id} from database")
        except Exception as e:
            logger.error(f"[TaskQueue] Failed to cleanup task {task_id}: {e}")

    def _cleanup_old_tasks(self):
        """Remove old completed tasks to prevent memory leaks"""
        cutoff_time = time.time() - self.max_task_age
        
        with self._lock:
            to_remove = []
            for task_id, task_result in self.task_results.items():
                if (task_result.status in [TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED] and
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

        # Create task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatusEnum.PENDING,
            created_at=time.time()
        )

        with self._lock:
            self.task_results[task_id] = task_result
            self.metrics['tasks_submitted'] += 1

        # CRITICAL FIX: Create DB task SYNCHRONOUSLY first
        if self.enable_db_persistence:
            db_created = False
            try:
                if self._async_loop and not self._async_loop.is_closed():
                    future = asyncio.run_coroutine_threadsafe(
                        self._create_db_task(task_id, func, args, kwargs, config),
                        self._async_loop
                    )
                    # Wait for DB creation to complete before proceeding
                    future.result(timeout=10.0)  # 10 second timeout
                    db_created = True
                    logger.info(f"[TaskQueue] DB task {task_id} created synchronously")
                else:
                    logger.error(f"[TaskQueue] No event loop available for DB task {task_id}")
                    
            except asyncio.TimeoutError:
                logger.error(f"[TaskQueue] Timeout creating DB task {task_id}")
            except Exception as e:
                logger.error(f"[TaskQueue] Failed to create DB task {task_id}: {e}", exc_info=True)
            
            # If DB creation failed, we can still continue with in-memory tracking
            if not db_created:
                logger.warning(f"[TaskQueue] Continuing with in-memory only for task {task_id}")

        # Submit to queue AFTER DB creation
        try:
            self.queue.put((config.priority.value, task_id, func, args, kwargs, config), timeout=5.0)
            logger.info(f"[TaskQueue] Task {task_id} added to queue. Current queue size: {self.queue.qsize()}")
            logger.info(f"[TaskQueue] Task submitted: {task_id} ({func.__name__}) priority={config.priority.name}")
            return task_id
        except queue.Full:
            # If queue is full, clean up the DB record we just created
            if self.enable_db_persistence:
                self._queue_db_operation(functools.partial(self._cleanup_failed_task, task_id))
            
            # Also clean up in-memory tracking
            with self._lock:
                self.task_results.pop(task_id, None)
                self.metrics['tasks_submitted'] -= 1
            
            error_msg = f"[TaskQueue] Task queue is full! Failed to submit task {task_id}. Max queue size: {self.max_queue_size}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from None
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        with self._lock:
            if task_id not in self.task_results:
                return False

            task_result = self.task_results[task_id]
            
            if task_result.status in [TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]:
                return False  # Already completed

            self.cancelled_tasks.add(task_id)
            
            if task_result.status == TaskStatusEnum.PENDING:
                # If pending, update status immediately
                self._queue_db_operation(functools.partial(
                    self._update_db_status, task_id, TaskStatusEnum.CANCELLED
                ))
                self.metrics['tasks_cancelled'] += 1

            logger.info(f"[TaskQueue] Task {task_id} marked for cancellation")
            return True

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status and result"""
        with self._lock:
            task_result = self.task_results.get(task_id)
            return task_result.to_dict() if task_result else None

    def get_all_tasks(self, status_filter: Optional[TaskStatusEnum] = None) -> Dict[str, Dict[str, Any]]:
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
                'db_queue_size': self._db_queue.qsize() if self.enable_db_persistence else 0,
                'metrics': {
                    **self.metrics,
                    'average_execution_time': round(avg_execution_time, 2)
                },
                'configuration': {
                    'num_workers': self.num_workers,
                    'max_queue_size': self.max_queue_size,
                    'cleanup_interval': self.cleanup_interval,
                    'max_task_age': self.max_task_age,
                    'db_persistence_enabled': self.enable_db_persistence
                }
            }

    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Wait for a task to complete"""
        start_time = time.time()
        
        while True:
            task_status = self.get_task_status(task_id)
            if not task_status:
                return None

            if task_status['status'] in [TaskStatusEnum.SUCCESS, TaskStatusEnum.FAILED, TaskStatusEnum.CANCELLED]:
                return task_status

            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")

            time.sleep(0.1)  # Poll every 100ms

    def shutdown(self, timeout: float = 30.0) -> bool:
        """Gracefully shutdown the task queue"""
        logger.info("[TaskQueue] Initiating shutdown...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Stop async loop
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        
        # Wait for workers to finish
        start_time = time.time()
        for worker in self.workers:
            remaining_time = max(0, timeout - (time.time() - start_time))
            worker.join(timeout=remaining_time)
            
            if worker.is_alive():
                logger.warning(f"[TaskQueue] Worker {worker.name} did not shutdown gracefully")

        # Wait for DB worker
        if self._db_worker_thread and self._db_worker_thread.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self._db_worker_thread.join(timeout=remaining_time)

        # Wait for cleanup thread
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self._cleanup_thread.join(timeout=remaining_time)

        # Wait for async loop thread
        if self._loop_thread and self._loop_thread.is_alive():
            remaining_time = max(0, timeout - (time.time() - start_time))
            self._loop_thread.join(timeout=remaining_time)

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
    return TaskConfig(priority=TaskPriorityEnum.HIGH, timeout=timeout, max_retries=1)

def create_reliable_config(max_retries: int = 5) -> TaskConfig:
    return TaskConfig(max_retries=max_retries, retry_delay=2.0, timeout=600)

def create_quick_config() -> TaskConfig:
    return TaskConfig(timeout=30, max_retries=1, priority=TaskPriorityEnum.NORMAL)
