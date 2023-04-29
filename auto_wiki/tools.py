from __future__ import annotations

from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Any
from pydantic import Field

import asyncio
import nest_asyncio
import os
from contextlib import contextmanager
import pandas as pd
from typing import Optional

from langchain.agents import tool
from langchain.agents.agent_toolkits.pandas.base import create_pandas_dataframe_agent
from langchain.chains.combine_documents.refine import RefineDocumentsChain
from langchain.chains.qa_with_sources.loading import BaseCombineDocumentsChain
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.requests import TextRequestsWrapper
from langchain.llms.base import BaseLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import BaseTool
from langchain.vectorstores.base import VectorStore

from topic_refine_chain import TopicRefine, InitialSummaryChain, ContinueSummaryChain


async def async_load_playwright(url: str) -> str:
    """Load the specified URLs using Playwright and parse using BeautifulSoup."""
    from bs4 import BeautifulSoup
    from playwright.async_api import async_playwright

    results = ""
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()
            await page.goto(url)

            page_source = await page.content()
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            results = "\n".join(chunk for chunk in chunks if chunk)
        except Exception as e:
            results = f"Error: {e}"
        await browser.close()
    return results


def run_async(coro):
    event_loop = asyncio.get_event_loop()
    return event_loop.run_until_complete(coro)


@tool
def browse_web_page(url: str) -> str:
    """Verbose way to scrape a whole webpage. Likely to cause issues parsing."""
    return run_async(async_load_playwright(url))


def _get_text_splitter():
    return RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=40,
        length_function=len,
    )


class WebpageQATool(BaseTool):
    name = "query_webpage"
    description = (
        "Browse a webpage and retrieve the information relevant to the question."
    )
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    qa_chain: BaseCombineDocumentsChain

    def _run(self, url: str, question: str) -> str:
        """Useful for browsing websites and scraping the text information."""
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        web_docs = self.text_splitter.split_documents(docs)
        results = []
        # TODO: Handle this with a MapReduceChain
        for i in range(0, len(web_docs), 2):
            input_docs = web_docs[i : i + 2]
            window_result = self.qa_chain(
                {"input_documents": input_docs, "question": question},
                return_only_outputs=True,
            )
            results.append(f"Response from window {i} - {window_result}")
        results_docs = [
            Document(page_content="\n".join(results), metadata={"source": url})
        ]
        return self.qa_chain(
            {"input_documents": results_docs, "question": question},
            return_only_outputs=True,
        )

    async def _arun(self, url: str, question: str) -> str:
        raise NotImplementedError


class MemorizeTool(BaseTool):
    """TODO: check for duplicates"""

    name = "memorize"
    description = "Read all text from a url and store in memory database. Use this first so information is loaded into memory!"
    text_splitter: RecursiveCharacterTextSplitter = Field(
        default_factory=_get_text_splitter
    )
    memory: VectorStore

    def _run(self, url: str) -> str:
        result = browse_web_page.run(url)
        docs = [Document(page_content=result, metadata={"source": url})]
        docs = self.text_splitter.split_documents(docs)
        self.memory.add_documents(docs)

        return "Done."

    def _arun(self, url: str) -> str:
        raise NotImplementedError


class RecallTool(BaseTool):
    name: str = "recall"
    description: str = (
        "Use this to recall information from memory database."
        " It contains information about many papers,"
        " so it may not be specific to a single paper."
        " Input is the information to recall."
    )
    memory: VectorStore
    refine: RefineDocumentsChain

    def _run(self, to_recall: str) -> str:
        docs = self.memory.similarity_search(to_recall, k=8)
        result = self.refine.combine_docs(topic=to_recall, docs=docs)
        return result[0]

    async def _arun(self, to_recall: str) -> str:
        raise NotImplementedError

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, memory: VectorStore, return_intermediate_steps: bool = False
    ) -> RecallTool:
        init_refine = InitialSummaryChain.from_llm(llm=llm)
        continue_refine = ContinueSummaryChain.from_llm(llm=llm)

        refine = RefineDocumentsChain(
            initial_llm_chain=init_refine,
            refine_llm_chain=continue_refine,
            document_variable_name="document",
            initial_response_name="summary",
            return_intermediate_steps=return_intermediate_steps,
        )

        return cls(memory=memory, refine=refine)


class UrlSummaryTool(BaseTool):
    """A tool for summarizing a website for specific information."""

    name = "url-summary"
    description = (
        "Use this when you need to summarize a website for specific information."
        " This tool generates more specific information than a web search."
        ' Input is json with two keys: "url" and "topic".'
        ' "url" is the url of the website and "data" is the topic you want to summarize.'
        # NOTE: These examples are difficult to include in the prompt due to the curly braces.
        # For example `{"url":"https://www.python.com","topic","main function"}` will search python.com for information about the main function.
        # Another example `{"url":"https://www.reddit.com/","topic":"top post"}` will search reddit for the top post."""
    )

    refine_chain: TopicRefine
    requests_wrapper: TextRequestsWrapper

    @classmethod
    def from_llm(
        cls,
        llm: BaseLLM,
        chunk_size: int = 3000,
        chunk_overlap: int = 100,
    ) -> BaseTool:
        """Create a tool from an LLM."""
        refine_chain = TopicRefine.from_llm(
            llm, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        requests_wrapper = TextRequestsWrapper()

        tool = cls(refine_chain=refine_chain, requests_wrapper=requests_wrapper)

        return tool

    # This is for Auto-GPT where it always gives a dict
    def _run(self, url: str, topic: str) -> str:
        try:
            page = self.requests_wrapper.get(url)
        except:
            return "Unable to load website."

        page_text = BeautifulSoup(page, "html.parser").get_text()
        return self.refine_chain.run(**{"document": page_text, "topic": topic})

    async def _arun(self, input: str) -> str:
        raise NotImplementedError
