"""OFFLINE stub for the ECS client.

In the offline build LOCAL_DEV_MODE is always on, so _build_and_launch_eval takes
the local-subprocess branch (_launch_local_eval) and these functions are never
called. They raise loudly if something is misconfigured, rather than failing deep
inside boto3.
"""
import os

# Kept as harmless exported constants for compatibility.
ECS_CLUSTER = os.environ.get('ECS_CLUSTER', '360eval-cluster')
EVAL_TASK_DEFINITION = os.environ.get('EVAL_TASK_DEFINITION', '360eval-worker')
EVAL_WORKER_CONTAINER = os.environ.get('EVAL_WORKER_CONTAINER', '360eval-worker')

_OFFLINE_MSG = (
    'ECS is not used in the offline build. Set LOCAL_DEV_MODE=true so evaluations '
    'run as a local subprocess.'
)


def launch_eval_task(*args, **kwargs):
    raise NotImplementedError(_OFFLINE_MSG)


def get_task_status(*args, **kwargs):
    raise NotImplementedError(_OFFLINE_MSG)
