#
# This file is part of rubin_rag.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Set up a Streamlit-based chatbot using Weaviate for vector search and
GPT-4o-mini for answering user queries.
"""

import os

import streamlit as st
from custom_weaviate_vector_store import CustomWeaviateVectorStore
from langchain_community.chat_message_histories import (
    StreamlitChatMessageHistory,
)
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_callback import get_streamlit_cb
from utils import load_config
from weaviate.classes.query import Filter
from weaviate.client import WeaviateClient

from rubin.rag.ingestion_pipeline.ingestor.client import connect

_config = load_config()


def submit_text() -> None:
    """Submit the user input."""
    st.session_state.message_sent = True


@st.cache_resource(ttl="1h")
def configure_client() -> WeaviateClient:
    """Configure the Weaviate client.

    Delegates to the ingestion pipeline's connector, which routes on
    ``config["weaviate"]["connection_mode"]`` (custom / weaviate_cloud /
    local) and applies ``WEAVIATE_API_KEY`` as each mode requires.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    return connect(_config, headers={"X-OpenAI-Api-Key": openai_api_key})


def configure_retriever() -> VectorStoreRetriever:
    """Configure the Weaviate retriever."""
    selected_sources = [
        source.lower() for source in st.session_state["required_sources"]
    ]
    if selected_sources:
        filters = Filter.by_property("source_key").contains_any(
            selected_sources
        )
    search_kwargs = {"k": 6, "where_filter": filters}

    return CustomWeaviateVectorStore(
        client=configure_client(),
        index_name=_config["weaviate"]["collection"],
        text_key="page_content",
        # NOTE (temporary): the query-side embedder is hardcoded to LangChain's
        # OpenAIEmbeddings. Fine for now - we only use OpenAI and are just
        # keeping LangChain current. A middle step before dropping LangChain:
        # map config["embedding"]["provider"] to the matching LangChain class
        # (OpenAIEmbeddings / CohereEmbeddings / ...), so the provider becomes
        # config-driven rather than hardcoded. Eventually replaced entirely by
        # the pipeline's provider-agnostic embedder
        # (ingestion_pipeline/ingestor/embedder.py). Left as a comment for now.
        embedding=OpenAIEmbeddings(
            model=_config["embedding"]["model"],
            dimensions=_config["embedding"]["dimensions"],
        ),
        embedding_config=_config["embedding"],
        attributes=["source", "source_key"],  # Metadata to fetch
    ).as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


def _format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved documents into the prompt's {context} block."""
    return "\n\n".join(doc.page_content for doc in docs)


def create_qa_chain(
    retriever: VectorStoreRetriever,
) -> Runnable:
    """Create a QA chain for the chatbot."""
    # Setup ChatOpenAI (Language Model)
    llm = ChatOpenAI(
        model=_config["llm"]["model"], temperature=0, streaming=True
    )

    # Define the system message template
    system_template = """You are Rubin AI Assistant, a helpful assistant at
    Vera C Rubin Observatory.
    Do your best to answer the questions in as much detail as possible.
    Do not attempt to provide an answer if you do not know the answer.
    In your response, do not recommend reading elsewhere.
    Use the following pieces of context to answer the user's
    question at the end.
    ----------------
    {context}
    ----------------"""

    # Create a ChatPromptTemplate for the QA conversation
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_template),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("Question:```{input}```"),
        ]
    )

    # Answer sub-chain: build the prompt's input map from the payload —
    # `context` is formatted to text here while the outer `context` document
    # list is left untouched for the UI — then call the LLM and parse to text.
    answer_chain: Runnable = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "context": lambda x: _format_docs(x["context"]),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # Full chain: retrieve on `input`, keep the Document list in `context`,
    # then attach the generated `answer`. Output mirrors the previous
    # create_retrieval_chain shape:
    # {"input", "chat_history", "context": list[Document], "answer": str}.
    return RunnablePassthrough.assign(
        context=lambda x: retriever.invoke(x["input"])
    ) | RunnablePassthrough.assign(answer=answer_chain)


def handle_user_input(
    qa_chain: Runnable, msgs: StreamlitChatMessageHistory
) -> None:
    """Handle user input and chat history."""
    # Check if the message history is empty or the user
    # clicked the "Clear message history" button
    if len(msgs.messages) == 0:
        msgs.clear()

    # Define avatars for user and assistant messages
    avatars = {"human": "user", "ai": "assistant"}
    avatar_images = {
        "human": "./static/user_avatar.png",
        "ai": "./static/rubin_telescope.png",
    }

    for msg in msgs.messages:
        st.chat_message(
            avatars[msg.type], avatar=avatar_images[msg.type]
        ).write(msg.content)

    # Handle new user input
    if user_query := st.chat_input(
        placeholder="Message Rubin AI", on_submit=submit_text
    ):
        with st.chat_message("user", avatar=avatar_images["human"]):
            st.write(user_query)
        msgs.add_user_message(user_query)

        with st.chat_message("assistant", avatar=avatar_images["ai"]):
            stream_handler = get_streamlit_cb(st.empty())

            # Invoke retriever logic
            result = qa_chain.invoke(
                {
                    "input": user_query,
                    "chat_history": msgs.messages,
                },
                {"callbacks": [stream_handler]},
            )
            msgs.add_ai_message(result["answer"])

            # Display source documents in an expander
            with st.expander("See sources"):
                scores = [
                    chunk.metadata["score"] for chunk in result["context"]
                ]

                if scores:
                    max_score = max(scores)
                    threshold = (
                        max_score * 0.9
                    )  # Set threshold to 90% of the highest score
                    cited_sources = set()

                    for chunk in result["context"]:
                        score = chunk.metadata["score"]

                        # Only show sources with scores
                        # above the threshold, skip duplicates
                        if score >= threshold:
                            source = chunk.metadata["source"]
                            if source not in cited_sources:
                                st.info(f"Source: {source}")
                                cited_sources.add(source)
