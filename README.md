In this project, I built an AI agent using LangChain, LangGraph, and a Retrieval-Augmented Generation (RAG) pipeline, and exposed the whole system through a FastAPI backend. The main idea was to create a simple but effective agent that can understand prompts, fetch relevant information when required, and give clear responses. The API exposes a single route (POST /run-agent), which makes it easy to test or integrate with other systems.

1. Use of Graph RAG / Agentic RAG

I followed an Agentic RAG design using LangGraph. Instead of always running retrieval first, the agent decides when to retrieve based on its reasoning. LangGraph lets me organize the workflow as a graph of nodes — reasoning, retrieval, and tool execution.

This approach improves the system in a couple of ways:

It handles multi-step reasoning better because it can think through the problem instead of replying immediately.

It provides higher recall and precision, because retrieval is triggered only when the LLM feels it needs more information.

This structure also makes debugging easier because I can see the exact flow of execution across nodes.

2. Knowledge Graph Integration

While I haven’t added a Knowledge Graph (KG) yet, the architecture is already designed in a way that a KG can be plugged in later. A KG would be especially helpful for structured domains like ad-tech. For example, I could model relationships such as:

Platforms → supported ad formats

User intent → recommended strategies

Creative type → best-performing platform

Example: If someone asks for “a performance ad for Gen Z on Instagram,” a KG could help map platform → audience insight → creative angle. This would make the responses more accurate, reduce hallucinations, and add domain consistency.

3. Evaluation Strategy (Using LanFuse)

For observability and evaluation, I used LanFuse throughout development. It allowed me to:

Trace every node execution in LangGraph

Inspect LLM thought process and intermediate reasoning

See which retrieval chunks were pulled

Detect hallucination or off-topic reasoning

Compare different versions of prompts easily

In terms of formal evaluation, I planned and tested with metrics like:

Relevance score

Hallucination rate

ROUGE / BLEU when generating summaries

F1 score for extraction-style prompts

API success rate to check FastAPI stability

LanFuse played a major role in debugging and improving the workflow as I built the agent.

4. Pattern Recognition & Improvement Loop (Future Possibility)

I have not implemented this part yet, but the agent can be extended to support continuous improvement. In the future, it could include:

LangGraph memory nodes for long-term context

A feedback loop where wrong outputs are logged and used to refine prompts

Retrieval re-ranking to boost relevant document selection

Automatic prompt optimization based on past responses

These additions would allow the agent to learn from past interactions without retraining the underlying model.