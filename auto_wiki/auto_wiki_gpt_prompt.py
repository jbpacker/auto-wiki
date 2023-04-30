import json
import time
from typing import Any, Callable, List

from pydantic import BaseModel

from langchain.prompts.chat import (
    BaseChatPromptTemplate,
)
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.tools.base import BaseTool
from langchain.vectorstores.base import VectorStoreRetriever

FINISH_NAME = "finish"


class PromptGenerator:
    """A class for generating custom prompt strings.

    Does this based on constraints, commands, resources, and performance evaluations.
    """

    def __init__(self) -> None:
        """Initialize the PromptGenerator object.

        Starts with empty lists of constraints, commands, resources,
        and performance evaluations.
        """
        self.commands: List[BaseTool] = []
        self.response_format = {
            "thoughts": {
                "text": "thought",
                "reasoning": "reasoning",
                "criticism": "constructive self-criticism",
                "speak": "thoughts summary to say to user",
            },
            "command": {"name": "command name", "args": {"arg name": "value"}},
        }

    def add_tool(self, tool: BaseTool) -> None:
        self.commands.append(tool)

    def _generate_command_string(self, tool: BaseTool) -> str:
        output = f"{tool.name}: {tool.description}"
        output += f", args json schema: {json.dumps(tool.args)}"
        return output

    def _generate_numbered_list(self, items: list, item_type: str = "list") -> str:
        """
        Generate a numbered list from given items based on the item_type.

        Args:
            items (list): A list of items to be numbered.
            item_type (str, optional): The type of items in the list.
                Defaults to 'list'.

        Returns:
            str: The formatted numbered list.
        """
        if item_type == "command":
            command_strings = [
                f"{i + 1}. {self._generate_command_string(item)}"
                for i, item in enumerate(items)
            ]
            finish_description = (
                "use this to signal that you have finished all your objectives"
            )
            finish_args = (
                '"response": "final response to let '
                'people know you have finished your objectives"'
            )
            finish_string = (
                f"{len(items) + 1}. {FINISH_NAME}: "
                f"{finish_description}, args: {finish_args}"
            )
            return "\n".join(command_strings + [finish_string])
        else:
            return "\n".join(f"{i+1}. {item}" for i, item in enumerate(items))

    def generate_prompt_string(self) -> str:
        """Generate a prompt string.

        Returns:
            str: The generated prompt string.
        """
        formatted_response_format = json.dumps(self.response_format, indent=4)
        prompt_string = (
            f"Commands:\n"
            f"{self._generate_numbered_list(self.commands, item_type='command')}\n\n"
            f"You should only respond in JSON format as described below "
            f"\nResponse Format: \n{formatted_response_format} "
            f"\nEnsure the response can be parsed by Python json.loads"
        )

        return prompt_string


def get_prompt(tools: List[BaseTool]) -> str:
    """This function generates a prompt string.

    It includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """

    # Initialize the PromptGenerator object
    prompt_generator = PromptGenerator()

    # Add commands to the PromptGenerator object
    for tool in tools:
        prompt_generator.add_tool(tool)

    # Generate the prompt string
    prompt_string = prompt_generator.generate_prompt_string()

    return prompt_string


class AutoWikiGPTPrompt(BaseChatPromptTemplate, BaseModel):
    tools: List[BaseTool]
    token_counter: Callable[[str], int]

    def construct_full_prompt(self, task: str, objective: str) -> str:
        prompt_start = (
            "You are a tool selection AI working with other AIs to add documentation to a wiki. "
            "Your job is to prompt and use commands to complete the current task.\n"
            "Put plenty of detailed information from your memory into the command args. "
            "It's highly important that the commands get detailed information.\n"
        )
        # Construct full prompt
        full_prompt = f"{prompt_start}\nULTIMATE OBJECTIVE:\n{objective}\n\n"
        full_prompt += f"CURRENT TASK:\n{task}\n\n"

        full_prompt += f"{get_prompt(self.tools)}"
        return full_prompt

    def format_messages(self, **kwargs: Any) -> List[BaseMessage]:
        task = kwargs["task"]
        objective = kwargs["objective"]
        memory: VectorStoreRetriever = kwargs["memory"]
        previous_messages = kwargs["messages"]

        final_message = SystemMessage(
            content="Create a command to do the CURRENT TASK. Remember to use the response format. Begin!\n"
        )

        # Step 1: Add initial prompt
        base_prompt = SystemMessage(
            content=self.construct_full_prompt(task=task, objective=objective)
        )
        time_prompt = SystemMessage(
            content=f"The current time and date is {time.strftime('%c')}"
        )
        used_tokens = self.token_counter(base_prompt.content) + self.token_counter(
            time_prompt.content
        )

        # Step 2: Add memory to prompt
        relevant_docs = memory.get_relevant_documents(
            task + objective + str(previous_messages[-10:])
        )
        relevant_memory = [d.page_content for d in relevant_docs]
        relevant_memory_tokens = sum(
            [self.token_counter(doc) for doc in relevant_memory]
        )
        while used_tokens + relevant_memory_tokens > 5500:
            relevant_memory = relevant_memory[:-1]
            relevant_memory_tokens = sum(
                [self.token_counter(doc) for doc in relevant_memory]
            )
        content_format = (
            f"This task reminds you of these events "
            f"from your past:\n{relevant_memory}\n\n"
        )
        memory_message = SystemMessage(content=content_format)
        used_tokens += self.token_counter(memory_message.content)
        used_tokens += self.token_counter(final_message.content)

        # Step 3: Add previous messages to prompt
        historical_messages: List[BaseMessage] = []
        message_tokens = 0
        for message in previous_messages[-10:][::-1]:
            message_tokens += self.token_counter(message.content)
            if used_tokens + message_tokens > 7200:
                break
            historical_messages = [message] + historical_messages

        # Step 4: Construct the prompt
        messages: List[BaseMessage] = [
            base_prompt,
            time_prompt,
            memory_message,
        ]
        messages += historical_messages
        messages.append(final_message)
        return messages
