"""Module with Celery configurations to Audio Length worker."""
from kombu import Queue

# Set worker to ack only when return or failing (unhandled expection)
task_acks_late = True

# Worker only gets one task at a time
worker_prefetch_multiplier = 1

# Create queue for worker
task_queues = [Queue(name="data_hprs")]

# Set Redis key TTL (Time to live)
result_expires = 60 * 60 * 48  # 48 hours in seconds
