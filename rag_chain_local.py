"""
Local RAG chain using Ollama — no API keys, no external calls.
Models used: mistral:latest (chat) + nomic-embed-text:latest (embeddings)
"""

from langchain_core.prompts import (
    ChatPromptTemplate, SystemMessagePromptTemplate,
    HumanMessagePromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
)
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from pydantic import BaseModel, Field

# 1. Output schema — what we want back, not just raw text
class RAGAnswer(BaseModel):
    answer: str = Field(description="The answer to the user's question")
    sources: list[str] = Field(description="Document IDs used")

parser = PydanticOutputParser(pydantic_object=RAGAnswer)

# 2. Embeddings — nomic-embed-text runs locally via Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 3. Example selector — pulls the most relevant few-shot examples
# for THIS query, from a bank of past Q&A pairs
examples = [
    {"input": "What is a vector store?",
     "output": '{"answer": "A database for embeddings...", "sources": ["doc_12"]}'},
    # ...more examples
]

# IMPORTANT: input_keys restricts the selector to comparing only the
# "input" field when searching for similar examples. Without this, it
# tries to join every key passed to chain.invoke() (context, chat_history,
# question, format_instructions) into one search string — and since
# chat_history is a list, that join fails with a TypeError.
example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, embeddings, FAISS, k=2, input_keys=["input"]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_selector=example_selector,
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

# 4. Chat prompt template — system instructions + retrieved context
# + history placeholder + few-shot examples + the live question
#
# NOTE: local models are much less reliable at strictly following a JSON
# schema than GPT-4/Claude. Being explicit and repeating the instruction
# in the human turn as well noticeably improves compliance with Mistral.
prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You answer using only the provided context.\nContext:\n{context}\n"
        "You MUST respond with valid JSON only, matching this format exactly:\n"
        "{format_instructions}\nDo not include any text outside the JSON object."
    ),
    few_shot_prompt,
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template(
        "{question}\n\n(Respond with JSON only, matching the required format.)"
    ),
])

# 5. Chat model — mistral:latest running locally via Ollama
llm = ChatOllama(model="mistral", temperature=0)

# 6. Assemble and run the chain
chain = prompt | llm | parser

if __name__ == "__main__":
    question = "How does the example selector pick examples?"

    result = chain.invoke({
        "context": "Paste or load your retrieved document text here.",
        "chat_history": [],
        "question": question,
        # "input" must match input_keys=["input"] on the example_selector above —
        # this is what the selector actually embeds and searches against.
        "input": question,
        "format_instructions": parser.get_format_instructions(),
    })

    print(result.answer)
    print(result.sources)