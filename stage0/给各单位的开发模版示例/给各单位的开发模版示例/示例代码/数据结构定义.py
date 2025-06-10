from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from drill.pipeline.fixed_size_buffer import FixedSizeBuffer


@dataclass
class ObsData:
    obs: Dict[str, Any]
    extra_info_dict: Dict[str, Any] = field(default_factory=dict)
    agent_name: str = ""


@dataclass
class ActionData:
    action: Dict[str, np.ndarray]    # action_head name to action
    predict_output: Dict[str, np.ndarray]
    action_mask: Dict[str, np.ndarray] = field(default_factory=dict)    # action_head name to mask
    agent_name: str = ""


class History:

    def __init__(self, maxlen):
        self._global = FixedSizeBuffer(maxlen)
        self._agents = defaultdict(lambda: FixedSizeBuffer(maxlen))

    @property
    def global_history(self):
        return self._global

    @property
    def agents_history(self) -> Dict[str, FixedSizeBuffer]:
        return self._agents

    def clear(self):
        self._global.clear()
        for agent_history in self._agents.values():
            agent_history.clear()
