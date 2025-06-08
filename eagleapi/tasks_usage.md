# BackgroundTaskQueue Usage Guide

This guide demonstrates how to use the `BackgroundTaskQueue` provided by Eagle for robust background task processing. This queue supports priorities, retries, timeouts, result tracking, cancellation, and more.

## 1. Importing the Task Queue

```python
from eagleapi import background_tasks
```

---

## 2. Submitting Tasks

You can submit any callable (function) to the queue. Optionally, you can provide configuration for retries, timeouts, and priority.

```python
def my_task(x, y):
    return x + y

# Submit a simple task
my_task_id = background_tasks.submit(my_task, 2, 3)

# Submit with custom configuration
from eagleapi.tasks.background import TaskConfig, TaskPriority
config = TaskConfig(max_retries=2, timeout=10, priority=TaskPriority.HIGH)
task_id = background_tasks.submit(my_task, 5, 7, config=config)
```

---

## 3. Task Priorities

Tasks can be prioritized using `TaskPriority`:

- `TaskPriority.LOW`
- `TaskPriority.NORMAL`
- `TaskPriority.HIGH`
- `TaskPriority.CRITICAL`

```python
from eagleapi.tasks.background import TaskPriority, TaskConfig
config = TaskConfig(priority=TaskPriority.CRITICAL)
task_id = background_tasks.submit(my_task, 1, 2, config=config)
```

---

## 4. Task Status and Results

You can check the status and result of a task using its ID:

```python
result = background_tasks.get_result(task_id)
if result:
    print(f"Status: {result.status}")
    print(f"Result: {result.result}")
    print(f"Error: {result.error}")
```

---

## 5. Retries and Timeouts

Configure how many times a task should retry and how long it can run:

```python
config = TaskConfig(max_retries=3, retry_delay=2.0, timeout=30)
task_id = background_tasks.submit(my_task, 4, 5, config=config)
```

---

## 6. Cancelling Tasks

You can cancel a pending or running task:

```python
background_tasks.cancel(task_id)
```

---

## 7. Waiting for Completion

You can block until a task completes (with an optional timeout):

```python
result = background_tasks.wait(task_id, timeout=20)
if result.status == 'success':
    print("Task finished successfully!")
```

---

## 8. Context Manager for Task Submission

Use the context manager to auto-submit and clean up:

```python
with background_tasks.task_context(my_task, 9, 10) as task_id:
    # Do something while the task runs
    ...
    # Optionally cancel or check status
```

---

## 9. Convenience Configurations

Use helper functions for common configs:

```python
from eagleapi.tasks.background import create_high_priority_config, create_reliable_config, create_quick_config

high_priority = create_high_priority_config(timeout=30)
reliable = create_reliable_config(max_retries=5)
quick = create_quick_config()

background_tasks.submit(my_task, 1, 1, config=high_priority)
```

---

## 10. Monitoring and Cleanup

- The queue automatically cleans up old task results.
- You can configure the number of workers and queue size when initializing `BackgroundTaskQueue` (see advanced usage in code).

---

## 11. Exception Handling

Custom exceptions are provided:
- `TaskTimeoutError`: Raised if a task exceeds its timeout.
- `TaskCancellationError`: Raised if a task is cancelled.

---

## 12. Example: Full Workflow

```python
from eagleapi import background_tasks
from eagleapi.tasks.background import TaskConfig, TaskTimeoutError

def slow_task(x):
    import time
    time.sleep(2)
    return x * 2

config = TaskConfig(timeout=5)
task_id = background_tasks.submit(slow_task, 10, config=config)
try:
    result = background_tasks.wait(task_id, timeout=6)
    print("Task result:", result.result)
except TaskTimeoutError:
    print("Task timed out!")
```

---

## 13. Production Deployment with Gunicorn

For production deployments, it's recommended to use Gunicorn as a process manager. Here's how to configure it for optimal performance with `BackgroundTaskQueue`.

### Basic Gunicorn Configuration

Create `gunicorn_config.py`:

```python
# gunicorn_config.py
import multiprocessing

# Number of worker processes
workers = multiprocessing.cpu_count() * 2 + 1

# Worker class (use Uvicorn's worker for ASGI)
worker_class = 'uvicorn.workers.UvicornWorker'

# Bind to localhost:8000
bind = '0.0.0.0:8000'

# Timeout (in seconds)
timeout = 120

# Keep-alive
keepalive = 5

# Logging
accesslog = '-'  # Log to stdout
errorlog = '-'   # Log to stderr
loglevel = 'info'
```

### Starting the Server

```bash
gunicorn -c gunicorn_config.py "eagleapi.main:create_app()"
```

### With Environment Variables

```bash
export WORKERS=$((2 * $(nproc) + 1))
export TIMEOUT=120
gunicorn -w $WORKERS -k uvicorn.workers.UvicornWorker -t $TIMEOUT --bind 0.0.0.0:8000 "eagleapi.main:create_app()"
```

### Systemd Service File

Create `/etc/systemd/system/eagleapi.service`:

```ini
[Unit]
Description=Eagle API Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/app
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/gunicorn -c gunicorn_config.py "eagleapi.main:create_app()"
Restart=always

[Install]
WantedBy=multi-user.target
```

### Managing the Service

```bash
# Reload systemd
dao reload

# Start the service
systemctl start eagleapi

# Enable on boot
systemctl enable eagleapi

# View logs
journalctl -u eagleapi -f
```

### Important Notes

1. **Worker Count**: 
   - For CPU-bound tasks: `workers = CPU cores + 1`
   - For I/O-bound tasks: `workers = (2 * CPU cores) + 1`

2. **Memory Management**:
   - Monitor memory usage per worker
   - Set `--max-requests` to restart workers after a number of requests
   - Use `--max-requests-jitter` to prevent all workers from restarting at once

3. **Timeouts**:
   - Set appropriate timeouts for your tasks
   - Consider using `--timeout` and `--graceful-timeout`

4. **Process Naming**:
   - Use `--name` to identify your workers in process lists

### Example with Max Requests

```bash
gunicorn \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --max-requests 1000 \
  --max-requests-jitter 50 \
  --timeout 120 \
  --name eagleapi \
  "eagleapi.main:create_app()"
```

## 14. Advanced: Customizing the Queue

You can create your own queue with custom worker count, queue size, etc.:

```python
from eagleapi.tasks.background import BackgroundTaskQueue
custom_queue = BackgroundTaskQueue(num_workers=8, max_queue_size=200)
```

---

## 14. Using BackgroundTaskQueue with FastAPI - Important Considerations

### In-Process vs Distributed

The `BackgroundTaskQueue` runs in the same process as your FastAPI application. This is simple but has important implications:

```python
# This runs in the same process as your web server
background_tasks = BackgroundTaskQueue(num_workers=4)
```

#### ✅ Benefits of In-Process:
- Simpler architecture (no extra services to manage)
- No network overhead for task submission
- Easier debugging (everything in one process)

#### ⚠️ Limitations to Consider:

1. **Shared Resources**:
   - All workers share the same process memory and CPU
   - Heavy tasks can impact web server performance
   - One task crashing could potentially crash the entire server

2. **Scaling**:
   - Limited to a single machine's resources
   - No built-in task persistence (tasks lost on server restart)
   - Hard to scale horizontally

3. **Deployment**:
   - Must be careful with worker count (don't exceed CPU cores)
   - Consider using a process manager (like Gunicorn with multiple workers)

### Recommended Production Setup

For production, consider these configurations:

```python
# In your app startup (e.g., main.py or app/__init__.py)
from eagleapi.tasks.background import BackgroundTaskQueue

# Limit workers to avoid starving the web server
# General rule: (total_cores - 1) for web, rest for workers
background_tasks = BackgroundTaskQueue(
    num_workers=max(1, os.cpu_count() - 1),  # Leave 1 core for web
    max_queue_size=1000  # Prevent memory issues
)

def get_background_tasks() -> BackgroundTaskQueue:
    return background_tasks
```

### When to Use a Dedicated Queue (Celery/ARQ/RQ)

Consider a distributed task queue when:
- Tasks are CPU-intensive (e.g., image processing, ML)
- You need task persistence across restarts
- You need to scale workers independently
- Tasks need to run on multiple machines

### Example: Hybrid Approach

```python
# For quick, in-memory tasks
from eagleapi import background_tasks

# For heavy or critical tasks
from celery import Celery
celery_app = Celery('tasks', broker='redis://localhost:6379/0')

@router.post("/process-image")
async def process_image():
    # Light task - use in-memory queue
    if task_is_light():
        task_id = background_tasks.submit(process_thumbnail, image)
    # Heavy task - use Celery
    else:
        result = process_image_task.delay(image)
        task_id = result.id
    return {"task_id": task_id}
```

## 15. Using BackgroundTaskQueue with FastAPI

You can easily integrate `BackgroundTaskQueue` into your FastAPI endpoints for robust background processing.

### Example 1: Submitting a Task from an Endpoint

```python
from fastapi import APIRouter, HTTPException
from eagleapi import background_tasks

router = APIRouter()

def long_running_task(data: int):
    import time
    time.sleep(5)
    return data * 10

@router.post("/start-task/")
def start_task(data: int):
    task_id = background_tasks.submit(long_running_task, data)
    return {"task_id": task_id}
```

### Example 2: Polling for Task Results

```python
@router.get("/task-result/{task_id}")
def get_task_result(task_id: str):
    result = background_tasks.get_result(task_id)
    if not result:
        raise HTTPException(status_code=404, detail="Task not found")
    return {
        "status": result.status.value,
        "result": result.result,
        "error": result.error,
        "started_at": result.started_at,
        "completed_at": result.completed_at
    }
```

### Example 3: Waiting for Task Completion in Endpoint (Not Recommended for Long Tasks)

```python
@router.post("/run-and-wait/")
def run_and_wait(data: int):
    task_id = background_tasks.submit(long_running_task, data)
    result = background_tasks.wait(task_id, timeout=10)
    return {"status": result.status.value, "result": result.result}
```

> **Why Submit + Poll is Better for Long-Running Tasks**
> 
> When handling long-running tasks in a web server like FastAPI, you have two main approaches:
> 
> **1. Blocking Approach (Not Recommended)**
> ```python
> @router.post("/run-and-wait/")  # ❌ Not ideal for long tasks
> def run_and_wait(data: int):
>     # This blocks the worker thread until completion
>     result = background_tasks.wait(task_id, timeout=30)
>     return {"result": result}
> ```
> - **Problem**: Each request ties up a worker thread while waiting
> - **Impact**: Under high load, your server could run out of available workers
> - **Result**: Other requests get queued or fail with timeouts
>
> **2. Submit + Poll Pattern (Recommended)**
> ```python
> @router.post("/start-task/")  # ✅ Better approach
> def start_task(data: int):
>     # Returns immediately
>     task_id = background_tasks.submit(long_task, data)
>     return {"task_id": task_id}
>
> @router.get("/task-result/{task_id}")
> def get_result(task_id: str):
>     # Client polls this endpoint
>     result = background_tasks.get_result(task_id)
>     return {"status": result.status, ...}
> ```
> - **Benefit**: Server resources aren't tied up waiting
> - **Scalability**: Handles many more concurrent users
> - **Resilience**: Survives client disconnections
> - **Progress**: Can implement progress tracking
>
> **When to Use Each**:
> - **Submit + Poll**: For tasks > 1 second
> - **Wait in Handler**: Only for very fast (<100ms) operations

> **Visualization**:
> ```
> [Client]                    [Server]                   [Worker Threads]
>    |                           |                                |
>    | POST /start-task         |                                |
>    |-------------------------->|                                |
>    |     {task_id: "abc123"}   |                                |
>    |<--------------------------|                                |
>    |                           |                                |
>    | GET /task-result/abc123  |                                |
>    |------------------------->|                                |
>    |     {status: "running"}   |                                |
>    |<-------------------------|                                |
>    |                           | [Worker processes task]         |
>    |                           |------------------------------->|
>    |                           |                                |
>    | GET /task-result/abc123  |                                |
>    |------------------------->|                                |
>    |     {status: "success"}   |                                |
>    |<-------------------------|                                |
> ```

---

## API Reference

See `eagleapi/tasks/background.py` for full API details and advanced features.
