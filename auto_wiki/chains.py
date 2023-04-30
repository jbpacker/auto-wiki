from typing import Dict, List, Optional, Any

from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM
from langchain.tools.base import BaseTool

from auto_wiki_gpt_prompt import AutoWikiGPTPrompt


def get_tool_prompt(tools: List[BaseTool]) -> str:
    """This function generates a prompt string.

    It includes various constraints, commands, resources, and performance evaluations.

    Returns:
        str: The generated prompt string.
    """
    prompt = ""

    for tool in tools:
        prompt += f"{tool.name}: {tool.description}\n"

    return prompt


class TodoChain(LLMChain):
    """Chain to generate a todo list."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, other_tools: List[BaseTool], verbose: bool = False
    ) -> LLMChain:
        todo_template = (
            "You are a planner who is an expert at coming up with a todo list for a given objective."
            "You have access to the following commands: \n"
            + get_tool_prompt(other_tools)
            + "Each todo item should use exactly one command."
            " Create a todo list for this objective: {objective}\n\n"
        )
        prompt = PromptTemplate(
            template=todo_template,
            input_variables=["objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskCreationChain(LLMChain):
    """Chain to generates tasks."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, tools: List[BaseTool], verbose: bool = False
    ) -> LLMChain:
        """Get the response parser."""
        task_creation_template = (
            "You are a task creation AI that uses the result of an execution agent"
            " to create new tasks with the following objective: \n{objective}\n"
            "You only have access to the following tools. Only create tasks that use these tools.\n"
            + get_tool_prompt(tools)
            + "Each todo item needs to use EXACTLY one command."
            " The last completed task has the result: \n{result}\n"
            " previously you've completed the following tasks: \n{completed_tasks}\n"
            " This result was based on this task description: \n{task_description}\n"
            " These are incomplete tasks: \n{incomplete_tasks}\n"
            " Based on the result, create new tasks to be completed"
            " by the AI system that do not overlap with incomplete tasks."
            " Return the tasks as a numbered list of short sentences."
        )
        prompt = PromptTemplate(
            template=task_creation_template,
            input_variables=[
                "result",
                "task_description",
                "completed_tasks",
                "incomplete_tasks",
                "objective",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskPrioritizationChain(LLMChain):
    """Chain to prioritize tasks."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, tools: List[BaseTool], verbose: bool = False
    ) -> LLMChain:
        """Get the response parser."""
        task_prioritization_template = (
            "You are a task prioritization AI tasked with cleaning the formatting of and reprioritizing"
            " the following tasks: \n{task_names}\n"
            "Previously you completed the following tasks: \n{completed_tasks}\n"
            " You only have access to the following tools. Only create tasks that use these tools.\n"
            + get_tool_prompt(tools)
            + "Each todo item needs to use EXACTLY one command. "
            "Consider the ultimate objective of your team: \n{objective}\n"
            "Incorporate all feedback from this critique of the task list: \n{critique}\n"
            "Keep the task list as short as possible by removing tasks that are irrelevant to the objective. tasksReturn the result as a numbered list, like:"
            " #. First task"
            " #. Second task"
            " Start the task list with number {next_task_id}."
        )
        prompt = PromptTemplate(
            template=task_prioritization_template,
            input_variables=[
                "task_names",
                "next_task_id",
                "completed_tasks",
                "objective",
                "critique",
            ],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class TaskCriticChain(LLMChain):
    """Chain to critique tasks."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, tools: List[BaseTool], verbose: bool = False
    ) -> LLMChain:
        task_critic_template = (
            "You are a critic AI that works with other AIs to complete an objective by using a task list."
            " As the critic you must give feedback about the task list in order to keep the list as small as possible and relevant to completing the ultimate objective."
            " You need to make sure all irrelevant tasks are removed."
            " For example, If the objective is to make a sandwich, then the task 'buy bread' is irrelevant."
            " If the objective is to write documentation, then the tasks to make a video or promote the documentation are irrelevant."
            " You only have access to the following tools. Make sure all tasks use these tools.\n"
            + get_tool_prompt(tools)
            + "Each todo item needs to use EXACTLY one command."
            " Only give feedback about the task list. Do not rewrite the task list."
            " The ultimate objective is:\n{objective}\n"
            "Previously you completed the following tasks: \n{completed_tasks}\n"
            "The new task list is: \n{task_names}\n"
            "Write a critique of the task list as a short paragraph. Include actions to improve the task list such as items to remove or combine. Be specific about which task as numbers may repeat."
        )
        prompt = PromptTemplate(
            template=task_critic_template,
            input_variables=["task_names", "completed_tasks", "objective"],
        )
        return cls(prompt=prompt, llm=llm, verbose=verbose)


class ToolSelectionChain(LLMChain):
    """Chain to select tools."""

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        tools: List[BaseTool],
        verbose: bool = False,
    ) -> LLMChain:
        prompt = AutoWikiGPTPrompt(
            tools=tools,
            input_variables=["memory", "objective", "task", "messages"],
            token_counter=llm.get_num_tokens,
        )
        return LLMChain(llm=llm, prompt=prompt, verbose=verbose)
