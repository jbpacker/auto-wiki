import argparse
from typing import Dict, List, Optional, Any
import pinecone
import faiss

from langchain.agents import load_tools, Tool
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.tools.arxiv.tool import ArxivQueryRun
from langchain.tools.file_management.read import ReadFileTool
from langchain.tools.file_management.list_dir import ListDirectoryTool
from langchain.tools.file_management.write import WriteFileTool
from langchain.utilities import ArxivAPIWrapper
from langchain.vectorstores.base import VectorStore
from langchain.vectorstores import FAISS, Pinecone
from wandb.integration.langchain import WandbTracer

from auto_wiki_gpt import AutoWikiGPT
from chains import TodoChain
from tools import WebpageQATool, MemorizeTool, RecallTool, UrlQueryTool
from utils import get_creds


def create_memory(in_memory: bool = True, keys: Dict[str, Any] = None) -> VectorStore:
    embd = OpenAIEmbeddings()
    embd_size = 1536
    if in_memory:
        index = faiss.IndexFlatL2(embd_size)
        store = FAISS(embd.embed_query, index, InMemoryDocstore({}), {})
    else:
        # Setup Pinecone
        assert (
            keys is not None and keys["pinecone_api_key"] is not None
        ), f"could not find pinecone api key in keys: {keys}"
        pinecone.init(
            api_key=keys["pinecone_api_key"],
            environment="asia-northeast1-gcp",
        )

        store = Pinecone(
            index=pinecone.Index("autowiki"),
            embedding_function=OpenAIEmbeddings().embed_query,
            text_key="text",
        )

    return store


def make_tools(memory: VectorStore):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, request_timeout=180)
    fast_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    tools = []
    # tools = load_tools(["google-search-results-json"], llm=fast_llm)
    # tools.append(WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)))
    tools.append(
        WriteFileTool(
            root_dir="./docs",
            description="Write to local file on disk. Does NOT work for urls or websites. Text MUST be formatted as a markdown file that's easy to read. Input: file_path: local file path, text: the text to write, append: to concatenate the text or replace it.",
        )
    )
    tools.append(
        ReadFileTool(
            root_dir="./docs",
            description="Read a single local file from disk. Does NOT work for urls or websites. Input: file path.",
        )
    )
    tools.append(
        ListDirectoryTool(
            root_dir="./docs",
            description="List files and directories in the wiki. Use this before reading or writing to files to understand the file structure. Input: file path to show.",
        )
    )
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper()))
    tools.append(MemorizeTool(memory=memory))
    tools.append(RecallTool.from_llm(llm=llm, memory=memory, verbose=False))
    tools.append(
        UrlQueryTool.from_llm(llm=fast_llm, chunk_size=3000, chunk_overlap=100)
    )

    # This one must be last to include the other tools in the prompt.
    tools.append(
        Tool(
            name="todo",
            func=TodoChain.from_llm(llm=llm, other_tools=tools).run,
            description="Use this to create a todo list. The input is the topic of the todo list, and this command will generate the list.",
        )
    )

    return tools


def main(arxiv_link: str):
    keys = get_creds()
    memory = create_memory(in_memory=True, keys=keys)
    tools = make_tools(memory)

    agent = AutoWikiGPT.from_llm_and_tools(
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0, request_timeout=180),
        vectorstore=memory.as_retriever(search_kwargs={"k": 8}),
        verbose=False,  # prints out prompts to models
    )

    prompt = (
        'Summarize this document and integrate it into the wiki by following the documentation template file "example.md." Document: '
        + arxiv_link
    )

    agent({"objective": prompt})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Auto-Wiki",
        description="Automatic wiki generation for research papers",
    )
    parser.add_argument(
        "arxiv_link",
        type=str,
        help="Link to an arxiv paper to summarize and add to the wiki.",
    )
    args = parser.parse_args()

    main(args.arxiv_link)
