from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pydantic import BaseModel, Extra, root_validator

from langchain.chains.base import Chain
from langchain.llms.base import BaseLLM
from langchain.text_splitter import TokenTextSplitter
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.schema import Document


class InitialSummaryChain(LLMChain):
    """Summarizes a document with respect to a topic."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        init_prompt_template = """DOCUMENT:
        {document}
        TOPIC:
        {topic}
        Summarize the document for information about the topic.
        If a summary for a document is desired, then summarize everything with a matching document for key points and a 1-2 sentence conclusion.
        Snippets that match the topic NEED TO BE quoted verbatim in the summary.
        If nothing in the doucment relates to the topic then write "None".
        SUMMARY:
        """

        init_prompt = PromptTemplate(
            template=init_prompt_template, input_variables=["topic", "document"]
        )

        return LLMChain(
            llm=llm, prompt=init_prompt, verbose=verbose, output_key="summary"
        )


class ContinueSummaryChain(LLMChain):
    """Continues to summarize by rolling a summary into a new one with a new document."""

    @classmethod
    def from_llm(cls, llm: BaseLLM, verbose: bool = False) -> LLMChain:
        cont_prompt_template = """DOCUMENT:
        {document}
        TOPIC:
        {topic}
        SUMMARY:
        {summary}
        You are combining many documents into a summary about a topic. Refine the summary into a combined summary by including new information about the topic in the document.
        If a summary for a document is desired, then summarize everything with a matching document for key points and a 1-2 sentence conclusion.
        Quoted snippets in the summary MUST be quoted verbatim in the combined summary.
        Snippets that matches the topic in the document NEED TO BE quoted verbatim in the summary.
        If there is nothing in the doucment about the topic then copy the summary, or repeat None if the summary is currently None.
        COMBINED SUMMARY:
        """
        cont_prompt = PromptTemplate(
            template=cont_prompt_template,
            input_variables=["topic", "document", "summary"],
        )
        return LLMChain(
            llm=llm, prompt=cont_prompt, verbose=verbose, output_key="summary"
        )


class TopicRefine(Chain, BaseModel):
    """
    Chain that takes website text (or anything long) and topic to refine into a concise summary.
    If the website is too long it splits it into smaller pieces to run a refine chain.
    Summary is generated with respect to the topic.
    """

    splitter: TokenTextSplitter
    refine_chain: RefineDocumentsChain
    return_intermediate_steps: bool

    @property
    def _chain_type(self) -> str:
        return "topic-refine"

    @property
    def input_keys(self) -> List[str]:
        return ["topic", "document"]

    @property
    def output_keys(self) -> List[str]:
        keys = ["summary"]
        if self.return_intermediate_steps:
            keys = keys + ["intermediate_steps"]

        return keys

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        return_intermediate_steps: bool = False,
        chunk_size: int = 3000,
        chunk_overlap: int = 100,
    ) -> TopicRefine:
        init_refine = InitialSummaryChain.from_llm(llm=llm)
        continue_refine = ContinueSummaryChain.from_llm(llm=llm)

        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        refine = RefineDocumentsChain(
            initial_llm_chain=init_refine,
            refine_llm_chain=continue_refine,
            document_variable_name="document",
            initial_response_name="summary",
            return_intermediate_steps=return_intermediate_steps,
        )

        return cls(
            splitter=splitter,
            refine_chain=refine,
            return_intermediate_steps=return_intermediate_steps,
        )

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        topic = inputs["topic"]
        document = inputs["document"]
        texts = [
            Document(page_content=t, metadata={"source": "None"})
            for t in self.splitter.split_text(document)
        ]

        summary = self.refine_chain.combine_docs(topic=topic, docs=texts)

        result: Dict[str, Any] = {"summary": summary[0]}

        if self.return_intermediate_steps:
            result["intermediate_steps"] = summary[1]["intermediate_steps"]

        return result
