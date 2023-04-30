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
from prompts import AI_NAME, AI_ROLE
from tools import WebpageQATool, MemorizeTool, RecallTool, UrlSummaryTool
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

    tools = load_tools(["google-search-results-json"], llm=fast_llm)
    # tools.append(WebpageQATool(qa_chain=load_qa_with_sources_chain(llm)))
    tools.append(WriteFileTool(root_dir="./docs"))
    tools.append(ReadFileTool(root_dir="./docs"))
    tools.append(ListDirectoryTool(root_dir="./docs"))
    tools.append(ArxivQueryRun(api_wrapper=ArxivAPIWrapper()))
    tools.append(MemorizeTool(memory=memory))
    tools.append(RecallTool.from_llm(llm=llm, memory=memory))
    tools.append(UrlSummaryTool.from_llm(llm=llm))

    # This one must be last to include the other tools in the prompt.
    tools.append(
        Tool(
            name="todo",
            func=TodoChain.from_llm(llm=llm, other_tools=tools).run,
            description="Use this to create a todo list. The input is the topic of the todo list, and this command will generate the list.",
        )
    )

    return tools


def main():
    keys = get_creds("auto_wiki/credentials.json")
    memory = create_memory(in_memory=True, keys=keys)
    tools = make_tools(memory)

    agent = AutoWikiGPT.from_llm_and_tools(
        tools=tools,
        llm=ChatOpenAI(model_name="gpt-4", temperature=0, request_timeout=180),
        vectorstore=memory.as_retriever(search_kwargs={"k": 8}),
        verbose=False,  # prints out prompts to models
    )

    document = "https://arxiv.org/pdf/2303.16199.pdf"

    prompt = (
        "summarize and integrate the following document into the resource wiki: "
        + document
    )

    agent({"objective": prompt})


if __name__ == "__main__":
    main()
