from typing import Type, Dict, List, Tuple
import numpy as np
import ray


class Worker(object):
    def __init__(self, args) -> None:
        super().__init__()
        