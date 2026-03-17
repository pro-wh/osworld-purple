import base64
import os
import sys

from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, FilePart, FileWithBytes, Message, TaskState, Part, TextPart
from a2a.utils import new_agent_text_message

from messenger import Messenger

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../osworld"))
from mm_agents.agent import PromptAgent

# Agent-config params: LLM tuning knobs owned by purple
MODEL = os.environ.get("MODEL", "gpt-4o")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1500"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.5"))
MAX_TRAJECTORY_LENGTH = int(os.environ.get("MAX_TRAJECTORY_LENGTH", "3"))
A11Y_TREE_MAX_TOKENS = int(os.environ.get("A11Y_TREE_MAX_TOKENS", "10000"))


class Agent:
    def __init__(self):
        self.messenger = Messenger()
        # Initialize other state here
        # prompt_agent is created on first step and kept alive across steps within a task
        self._prompt_agent: PromptAgent | None = None

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        """Implement your agent logic here.

        Args:
            message: The incoming message
            updater: Report progress (update_status) and results (add_artifact)

        Use self.messenger.talk_to_agent(message, url) to call other agents.
        """
        # Replace this example code with your agent logic

        await updater.update_status(
            TaskState.working, new_agent_text_message("Thinking...")
        )
        # Unpack parts sent by A2AClientAgent
        instruction = ""
        obs: dict = {}
        env_config: dict = {}

        for part in message.parts:
            root = part.root
            if isinstance(root, TextPart):
                instruction = root.text
            elif isinstance(root, FilePart):
                if isinstance(root.file, FileWithBytes):
                    obs["screenshot"] = base64.b64decode(root.file.bytes)
            elif isinstance(root, DataPart):
                if "env_config" in root.data:
                    env_config = root.data["env_config"]
                else:
                    obs.update(root.data)

        # Construct PromptAgent on first step of a task
        if self._prompt_agent is None:
            self._prompt_agent = PromptAgent(
                **env_config,
                model=MODEL,
                max_tokens=MAX_TOKENS,
                top_p=TOP_P,
                temperature=TEMPERATURE,
                max_trajectory_length=MAX_TRAJECTORY_LENGTH,
                a11y_tree_max_tokens=A11Y_TREE_MAX_TOKENS,
            )

        assert self._prompt_agent is not None
        response, actions = self._prompt_agent.predict(instruction, obs)

        await updater.add_artifact(
            parts=[
                Part(root=TextPart(text=response or "")),
                Part(root=DataPart(data={"actions": actions or []})),
            ],
            name="prediction",
        )
