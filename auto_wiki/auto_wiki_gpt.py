from __future__ import annotations

from collections import deque
from typing import List, Optional, Dict, Any

from pydantic import ValidationError, BaseModel, Field

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models.base import BaseChatModel
from langchain.experimental.autonomous_agents.autogpt.output_parser import (
    AutoGPTOutputParser,
    BaseAutoGPTOutputParser,
)
from langchain.experimental.autonomous_agents.autogpt.prompt_generator import (
    FINISH_NAME,
)
from langchain.schema import (
    AIMessage,
    BaseMessage,
    Document,
    HumanMessage,
    SystemMessage,
)
from langchain.llms import BaseLLM
from langchain.tools.base import BaseTool
from langchain.tools.human.tool import HumanInputRun
from langchain.vectorstores.base import VectorStoreRetriever

from chains import (
    TodoChain,
    TaskCriticChain,
    TaskPrioritizationChain,
    TaskCreationChain,
    ToolSelectionChain,
)


class AutoWikiGPT(Chain, BaseModel):
    """Agent class for interacting with Auto-GPT."""

    vectorstore: VectorStoreRetriever
    task_creation_chain: LLMChain
    task_critic_chain: LLMChain
    task_prioritization_chain: LLMChain
    tool_selection_chain: LLMChain
    output_parser: BaseAutoGPTOutputParser
    tools: List[BaseTool]
    task_id_counter: int = Field(1)
    feedback_tool: Optional[HumanInputRun] = None
    task_list: deque = Field(default_factory=deque)
    completed_tasks: List[Any] = Field(default_factory=list)
    full_message_history: List[BaseMessage] = Field(default_factory=list)
    verbose: bool = Field(False)

    @classmethod
    def from_llm_and_tools(
        cls,
        vectorstore: VectorStoreRetriever,
        tools: List[BaseTool],
        llm: BaseChatModel,
        human_in_the_loop: bool = False,
        output_parser: Optional[BaseAutoGPTOutputParser] = None,
        verbose: bool = False,
    ) -> AutoWikiGPT:
        human_feedback_tool = HumanInputRun() if human_in_the_loop else None

        task_creation_chain = TaskCreationChain.from_llm(
            llm=llm, tools=tools, verbose=verbose
        )
        task_critic_chain = TaskCriticChain.from_llm(
            llm=llm, tools=tools, verbose=verbose
        )
        task_prioritization_chain = TaskPrioritizationChain.from_llm(
            llm=llm, tools=tools, verbose=verbose
        )
        tool_selection_chain = ToolSelectionChain.from_llm(
            llm=llm, tools=tools, verbose=verbose
        )

        return cls(
            vectorstore=vectorstore,
            task_creation_chain=task_creation_chain,
            task_critic_chain=task_critic_chain,
            task_prioritization_chain=task_prioritization_chain,
            tool_selection_chain=tool_selection_chain,
            output_parser=output_parser or AutoGPTOutputParser(),
            tools=tools,
            feedback_tool=human_feedback_tool,
            verbose=verbose,
        )

    def print_task_list(self, prioritized: bool = True):
        if prioritized:
            print("\033[95m\033[1m\n*****TASK LIST*****\n\033[0m\033[0m")
        else:
            print("\033[95m\033[1m\n*****UNORDERED TASK LIST*****\n\033[0m\033[0m")

        for t in self.task_list:
            print(str(t["task_id"]) + ": " + t["task_name"])

    def print_next_task(self, task: Dict):
        print("\033[92m\033[1m" + "\n*****NEXT TASK*****\n" + "\033[0m\033[0m")
        print(str(task["task_id"]) + ": " + task["task_name"])

    def print_critic_result(self, critic_output: str):
        print("\033[93m\033[1m" + "\n*****CRITIC RESULT*****\n" + "\033[0m\033[0m")
        print(critic_output)

    def print_tool_selection(self, result: str):
        print("\033[93m\033[1m" + "\n*****TOOL REQUEST*****\n" + "\033[0m\033[0m")
        print(result)

    def print_task_result(self, result: str):
        print("\033[93m\033[1m" + "\n*****TASK RESULT*****\n" + "\033[0m\033[0m")
        print(result)

    def print_tool_selection_prompt(self, task, objective):
        print()
        print(
            self.tool_selection_chain.prompt.format(
                **{
                    "task": task["task_name"],
                    "objective": objective,
                    "messages": self.full_message_history,
                    "memory": self.vectorstore,
                }
            )
        )
        print()

    @property
    def input_keys(self) -> List[str]:
        return ["objective"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def get_new_tasks(
        self,
        objective: str,
        previous_task: str,
        completed_tasks: str,
        prev_task_result: str,
    ) -> List[Dict]:
        incomplete_tasks = ", ".join([t["task_name"] for t in self.task_list])
        new_tasks = self.task_creation_chain.run(
            result=prev_task_result,
            task_description=previous_task,
            completed_tasks=completed_tasks,
            incomplete_tasks=incomplete_tasks,
            objective=objective,
        )
        new_tasks = new_tasks.split("\n")
        return [
            {"task_name": task_name} for task_name in new_tasks if task_name.strip()
        ]

    def prioritize_tasks(
        self,
        this_task_id: int,
        completed_tasks: str,
        critique: str,
        objective: str,
    ) -> List[Dict]:
        """Prioritize tasks."""
        task_names = [t["task_name"] for t in list(self.task_list)]
        next_task_id = int(this_task_id) + 1
        response = self.task_prioritization_chain.run(
            task_names=task_names,
            next_task_id=next_task_id,
            completed_tasks=completed_tasks,
            critique=critique,
            objective=objective,
        )
        new_tasks = response.split("\n")
        prioritized_task_list = []
        for task_string in new_tasks:
            if not task_string.strip():
                continue
            task_parts = task_string.strip().split(".", 1)
            if len(task_parts) == 2:
                task_id = task_parts[0].strip()
                task_name = task_parts[1].strip()
                prioritized_task_list.append(
                    {"task_id": task_id, "task_name": task_name}
                )
        return prioritized_task_list

    def add_task(self, task: Dict):
        self.task_list.append(task)

    def _call(self, inputs: Dict[str:Any]) -> str:
        objective = inputs["objective"]
        self.add_task({"task_id": 1, "task_name": "Create a todo list."})

        # Interaction Loop
        loop_count = 0
        while True:
            self.print_task_list()

            # Step 1: Pull the first task
            task = self.task_list.popleft()
            self.print_next_task(task)

            # Step 2: Select an action to take
            # self.print_tool_selection_prompt(task, objective)
            assistant_reply = self.tool_selection_chain.run(
                task=task["task_name"],
                objective=objective,
                messages=self.full_message_history,
                memory=self.vectorstore,
            )
            this_task_id = int(task["task_id"])
            self.print_tool_selection(assistant_reply)

            # Step 3: Execute the action
            # Get command name and arguments
            action = self.output_parser.parse(assistant_reply)
            tools = {t.name: t for t in self.tools}
            if action.name == FINISH_NAME:
                return action.args["response"]
            if action.name in tools:
                tool = tools[action.name]
                try:
                    observation = tool.run(action.args)
                except ValidationError as e:
                    observation = (
                        f"Validation Error in args: {str(e)}, args: {action.args}"
                    )
                except Exception as e:
                    observation = (
                        f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
                    )
                result = f"Command {tool.name} returned: {observation}"
            elif action.name == "ERROR":
                result = f"Error: {action.args}. "
            else:
                result = (
                    f"Unknown command '{action.name}'. "
                    f"Please refer to the 'COMMANDS' list for available "
                    f"commands and only respond in the specified JSON format."
                )
            self.print_task_result(result)
            self.completed_tasks.append(task)
            current_completed_tasks = ", ".join(
                [t["task_name"] for t in self.completed_tasks]
            )
            full_result = assistant_reply + "\n\n" + result

            # Step 4: Store the results
            result_id = f'result_{task["task_id"]}'
            self.vectorstore.add_documents(
                [
                    Document(
                        page_content=full_result,
                        metadata={"task": task["task_name"]},
                        ids=result_id,
                    )
                ]
            )
            # Step 5: Create new tasks and reprioritize task list
            new_tasks = self.get_new_tasks(
                objective, task["task_name"], current_completed_tasks, full_result
            )
            for new_task in new_tasks:
                self.task_id_counter += 1
                new_task.update({"task_id": self.task_id_counter})
                self.add_task(new_task)
            self.print_task_list(prioritized=False)

            # Step 6: Criticize the results
            critique = self.task_critic_chain.run(
                task_names=", ".join([t["task_name"] for t in self.task_list]),
                completed_tasks=current_completed_tasks,
                objective=objective,
            )
            self.print_critic_result(critique)

            # Step 7: Reprioritize the task list
            self.task_list = deque(
                self.prioritize_tasks(
                    this_task_id=this_task_id,
                    completed_tasks=current_completed_tasks,
                    critique=critique,
                    objective=objective,
                )
            )

            memory_to_add = (
                f"Assistant Reply: {assistant_reply} " f"\nResult: {result} "
            )
            if self.feedback_tool is not None:
                feedback = f"\n{self.feedback_tool.run('Input: ')}"
                if feedback in {"q", "stop"}:
                    print("EXITING")
                    return "EXITING"
                memory_to_add += feedback

            self.vectorstore.add_documents([Document(page_content=memory_to_add)])
            self.full_message_history.append(SystemMessage(content=result))

            # Discontinue if continuous limit is reached
            loop_count += 1
