import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq         
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
model = ChatGroq( model="llama-3.1-8b-instant",
                  temperature=0,
                  api_key=GROQ_API_KEY)






from langchain_community.document_loaders import TextLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_PATH = os.path.join(BASE_DIR, "rag_docs", "marketing_blogs.txt")

loader = TextLoader(DOC_PATH, encoding="utf-8")

docs = loader.load()


from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
chunks = splitter.split_documents(docs)

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

PERSIST_DIR = "vectorstore"

# This Automatically rebuild vectorstore if missing (Render)
if not os.path.exists(PERSIST_DIR) or len(os.listdir(PERSIST_DIR)) == 0:
    print("Vectorstore missing. Rebuilding...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=emb,
        persist_directory=PERSIST_DIR
    )
    print("Vectorstore rebuilt successfully.")
else:
    vectorstore = Chroma(
        embedding_function=emb,
        persist_directory=PERSIST_DIR
    )

vector_retriever = vectorstore.as_retriever()

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnablePassthrough

def load_summarize_chain(llm, chain_type="map_reduce"):
    if chain_type != "map_reduce":
        raise ValueError("Only map_reduce is supported in this custom version.")

    map_prompt = PromptTemplate.from_template(
        "Summarize this chunk:\n\n{text}"
    )

    reduce_prompt = PromptTemplate.from_template(
        "Combine all the chunk summaries into one final coherent summary:\n\n{text}"
    )

    map_chain = map_prompt | llm
    reduce_chain = reduce_prompt | llm

    summary_chain = (
        RunnableParallel(map_results=map_chain.map())
        | (lambda x: {"text": "\n".join(x["map_results"])})
        | reduce_chain
    )

    return summary_chain

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=GROQ_API_KEY
)

summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

from langchain_core.tools import tool

@tool
def summary_tool(query: str) -> str:
    """Summarization tool for summarizing the State of AI report."""
    result = summary_chain.run(chunks)
    return result

@tool
def vector_tool(query: str) -> str:
    """Vector search tool for retrieving specific context from the State of AI report."""
    docs = vector_retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])
    response = llm.invoke(f"Use the context below to answer:\n\n{context}\n\nQuestion: {query}")
    return response.content

def pick_tool(name: str):
    name = name.strip().lower()
    if "summary" in name:
        return summary_tool
    if "vector" in name:
        return vector_tool
    return vector_tool  # fallback

from langchain_core.prompts import PromptTemplate

router_prompt = PromptTemplate.from_template("""
You are a tool routing AI.

Select the best tool for answering the user query.

Tools:
1. summary_tool → when the user wants summarization
2. vector_tool → when the user wants retrieval, search, or information lookup
3.direct → for simple/general questions not needing retrieval

User question: {question}

Return ONLY the tool name.
""")

from langchain_core.runnables import RunnableSequence

router_chain = router_prompt | llm

# print("ROUTER OUTPUT →", router_to_name.invoke({"question": "test"}))
# print("TOOL OUTPUT →", name_to_tool.invoke({"question": "test"}))

#print("AFTER router_to_name:", router_to_name.invoke({"question": "test"}))

from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableMap
router_to_name = router_chain | RunnableLambda(
    lambda msg: msg.content.strip().lower()
)
def select_tool_func(d):
    tool_name = d["tool_name"]
    return {
        "tool": pick_tool(tool_name),   # RETURNS Tool object
        "question": d["question"]       # keep question
    }

name_to_tool = RunnableMap({
    "tool_name": router_to_name,
    "question": lambda d: d["question"]
}) | RunnableLambda(select_tool_func)

# 3. Execute selected tool
tool_chain = name_to_tool | RunnableLambda(
    lambda d: d["tool"].run(d["question"])
)

class QueryEngine:
    def __init__(self, chain):
        self.chain = chain

    def query(self, question):
        return self.chain.invoke({"question": question})

query_engine = QueryEngine(tool_chain)



# 1. Input/Query
class MarketingRAGState(TypedDict):
    question: str
    tool_name: str
    tool_result: str
    final_answer: str
    evaluation: str

from langchain_core.messages import HumanMessage

def router_node(state: MarketingRAGState):
    tool_name = router_to_name.invoke({"question": state["question"]})
    return {"tool_name": tool_name}
def tool_executor_node(state: MarketingRAGState):
    rag_output = tool_chain.invoke({"question": state["question"]})

    clean_answer = llm.invoke(
        f"""
You are a top-tier creative copywriter from a global ad agency.
Your job: produce **punchy, hard-hitting ad lines** based on retrieved marketing context.

Tone:
- short
- bold
- energetic
- high-impact
- confident
- punchline-style

Rules:
- NO generic ad copy (“brighten your summer”, "don't miss out")
- NO long sentences
- NO soft/weak words
- NO over-explaining
- MAX creative punch in MIN words
- Strong verbs, vivid contrast, rhythmic phrasing
- Every line must feel like a **headline that slaps**

Output format:

1. **Creative Strategy (1–2 bullets MAX — short + sharp)**
2. **Punchy Ad Lines (6–8 lines — extremely punchy)**
3. **Execution Sparks (2 bullets)**

Context (retrieved from marketing blogs)
{rag_output}

Question: {state['question']}
"""
    ).content

    return {"tool_result": clean_answer}
def answer_node(state: MarketingRAGState):
    polished = llm.invoke(
        f"""
Polish the following marketing answer for clarity, grammar, and impact.
Ensure it reads like a blog expert wrote it:

{state['tool_result']}
"""
    ).content
    return {"final_answer": polished}
def evaluator_node(state: MarketingRAGState):
    evaluation = llm.invoke(
        f"""
You are evaluating the answer given by the RAG system.

Question: {state['question']}
Answer: {state['tool_result']}

Evaluate ONLY:

1. Is the answer correct based on marketing blogs context?
2. Is it complete? (contains all important points)
3. Is it concise and well structured?

Respond ONLY with: "good" or "retry"
"""
    ).content.strip().lower()

    return {"evaluation": evaluation}
def direct_answer_node(state: MarketingRAGState):
    answer = llm.invoke(
        f"""
Answer the user's question directly in a concise and helpful way.

Question: {state['question']}
"""
    ).content

    return {"final_answer": answer}

graph = StateGraph(MarketingRAGState)

# 1. Add nodes
graph.add_node("router", router_node)
graph.add_node("direct_answer", direct_answer_node)
graph.add_node("tool_executor", tool_executor_node)
graph.add_node("evaluator", evaluator_node)   # MUST BE ADDED BEFORE EDGES
graph.add_node("answer", answer_node)

# 2. Entry point
graph.set_entry_point("router")

# 3. Standard flow
graph.add_edge("router", "tool_executor")
graph.add_edge("router", "direct_answer")
graph.add_conditional_edges(
    "router",
    lambda state: state["tool_name"],
    {
        "vector_tool": "tool_executor",
        "summary_tool": "tool_executor",
        "direct": "direct_answer"        # NEW DIRECT PATH
    }
)
graph.add_edge("direct_answer", END)
graph.add_edge("tool_executor", "evaluator")   # executor → evaluator

# 4. Conditional edges (LOOP!)
graph.add_conditional_edges(
    "evaluator",
    lambda state: state["evaluation"],
    {
        "retry": "tool_executor",    # loop back
        "good": "answer"             # go forward
    }
)

# 5. Final node leads to END
graph.add_edge("answer", END)

# 6. Compile
marketing_graph = graph.compile()

result = marketing_graph.invoke({
    "question": "Give me the best ad copy ideas for a summer sale"
}
    )

print(result["final_answer"])