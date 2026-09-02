# LANGCHAIN LEARNING & PRACTICES

> A hands-on learning journey through LangChain — from basic LLM interaction to RAG and agentic systems.

This repository is dedicated to **learning, understanding, experimenting, and practicing LangChain**.

The notebooks are arranged as a progressive learning path. Each module focuses on a specific LangChain concept and gradually builds toward more advanced topics such as **LCEL, conversation memory, RAG, document processing, tools, and agents**.

The goal is not just to learn how to write LangChain code, but to understand:

* What each LangChain component does
* Why it is needed
* How different components connect together
* How data flows through a LangChain application
* When to use one approach over another
* How simple chains gradually evolve into RAG and agentic applications

---

# Learning Path

The recommended learning flow is:

```text
LangChain Basics
      ↓
Prompt Templates
      ↓
Model Parameters
      ↓
Runnables
      ↓
Output Parsers
      ↓
Chain Execution Methods
      ↓
LCEL
      ↓
Conversation Memory
      ↓
Memory Optimization
      ↓
Sequential & Conditional Chains
      ↓
Frontend Integration
      ↓
RAG Fundamentals
      ↓
Document Loaders
      ↓
Document Splitting
      ↓
Agents & Tools
      ↓
Advanced Agentic RAG
```

---

# 1. LangChain Basics

### Module 1

Start by understanding the basic idea of **LangChain** and how an LLM application is connected together.

### Topics

* Introduction to LangChain
* Basic LangChain workflow
* LLM / Chat Model interaction
* Basic prompt → model flow
* Understanding the role of LangChain between the application and the model

### Goal

Understand the basic idea of LangChain before moving into its individual components.

> 📘 **Deep Dive — Why not just call the model API directly?**
> A raw API call only gives you a single request → response cycle. LangChain adds a standard layer on top of that call — prompt templating, memory, retrieval, tool use, and output parsing — so those pieces can be swapped and recombined without rewriting the underlying application logic. Think of the LLM as the "engine" and LangChain as the "chassis" that lets you attach different components (prompts, memory, tools) around it in a consistent way.
>
> **Minimal example:**
> ```python
> from langchain_openai import ChatOpenAI
>
> model = ChatOpenAI(model="gpt-4o-mini")
> response = model.invoke("What is LangChain?")
> print(response.content)
> ```
> This single call is the foundation every later module builds on top of.

---

# 2. Prompt Templates

### Module 2.1

Prompt templates are one of the first important abstractions to understand because they separate the **prompt structure** from the **input data**.

### Topics

* `PromptTemplate`
* `ChatPromptTemplate`
* Difference between `PromptTemplate` and `ChatPromptTemplate`
* Static prompts vs dynamic prompts
* Prompt variables
* Creating prompts from templates
* Connecting prompts with models
* Prompt chaining

### Goal

Understand how prompts are created, parameterized, and connected to LLMs.

> 📘 **Deep Dive — `PromptTemplate` vs `ChatPromptTemplate`**
> `PromptTemplate` produces a single plain-text string, which suits text-completion-style models. `ChatPromptTemplate` produces a *list of role-tagged messages* (system, human, AI), which matches how chat models actually expect their input. In practice almost all modern usage favors `ChatPromptTemplate` because most production models are chat-tuned.
>
> ```python
> from langchain_core.prompts import ChatPromptTemplate
>
> prompt = ChatPromptTemplate.from_messages([
>     ("system", "You are a helpful {domain} expert."),
>     ("human", "{question}")
> ])
>
> formatted = prompt.invoke({"domain": "cooking", "question": "How do I poach an egg?"})
> ```
> **Best practice:** keep the system message stable and put only the variable parts (`{question}`, `{domain}`) in the template — this makes prompts easier to test and reuse across chains.

---

# 3. Model Parameters

### Module 2.2

After learning how to create prompts, understand how model configuration affects the generated response.

### Topics

* Chat models
* Model parameters
* Model configuration
* Temperature
* Model-specific configuration
* Comparing model behavior with different settings
* Difference between text-oriented and chat-oriented model interaction

### Goal

Understand that the model is not just a function that receives text — its configuration also affects how it behaves.

> 📘 **Deep Dive — Common parameters at a glance**
>
> | Parameter | Effect | Typical use |
> |---|---|---|
> | `temperature` | Controls randomness (0 = deterministic, 1+ = creative) | Low for factual/QA tasks, higher for creative writing |
> | `max_tokens` | Caps the length of the generated response | Control cost and response size |
> | `top_p` | Nucleus sampling — restricts token pool by cumulative probability | Alternative/complement to temperature |
> | `stop` | Sequence(s) that stop generation early | Prevent runaway output, enforce format |
>
> A useful habit while learning: run the **same prompt** through a chain twice, once with `temperature=0` and once with `temperature=0.9`, and compare outputs side by side — this makes the effect of the parameter concrete rather than theoretical.

---

# 4. Runnables

### Module 3

Runnables are one of the most important foundations for understanding modern LangChain composition.

### Topics

* Callable vs Runnable
* Runnable concept
* `RunnablePassthrough`
* `RunnableLambda`
* `RunnableParallel`
* `.assign()`
* Passing data between runnables
* Running multiple operations in parallel
* Combining different runnables
* Building chains using runnables

### Important Concepts

#### `RunnablePassthrough`

Passes the original input forward without modifying it.

#### `RunnableLambda`

Converts a normal Python function into a Runnable so it can participate in a LangChain pipeline.

#### `RunnableParallel`

Runs multiple Runnable operations using the same input and combines their results.

#### `.assign()`

Allows additional information to be generated while keeping existing input data.

### Goal

Understand the fundamental building blocks used to construct LangChain pipelines.

> 📘 **Deep Dive — Seeing `RunnableParallel` in action**
> ```python
> from langchain_core.runnables import RunnableParallel, RunnableLambda
>
> chain = RunnableParallel(
>     upper=RunnableLambda(lambda x: x["text"].upper()),
>     length=RunnableLambda(lambda x: len(x["text"])),
> )
>
> chain.invoke({"text": "hello langchain"})
> # -> {"upper": "HELLO LANGCHAIN", "length": 16}
> ```
> **Why this matters:** every LangChain component — prompts, models, parsers, retrievers — implements the same `Runnable` interface (`.invoke()`, `.batch()`, `.stream()`). That's exactly what makes the `|` pipe operator in LCEL (Module 6) possible later on.

---

# 5. Output Parsers

### Module 4

LLMs normally return model responses, but applications often need the output in a specific format.

Output parsers help convert model output into a more useful structure.

### Topics

* Why output parsers are required
* `StrOutputParser`
* JSON output parsing
* `JsonOutputParser`
* Pydantic output parsing
* `PydanticOutputParser`
* Structured output parsing
* `StructuredOutputParser`
* `CommaSeparatedListOutputParser`
* `DatetimeOutputParser`
* `EnumOutputParser`
* Retry output parsing
* Output validation
* Pydantic schemas
* Format instructions

### Important Concept

Instead of simply using:

```python
response.content
```

an output parser provides a controlled way to transform the model response into the required format.

### Pydantic

Understand how Pydantic can be used to define the expected structure and validate model output.

### Goal

Learn how to move from **unstructured LLM output → structured application-ready output**.

> 📘 **Deep Dive — `PydanticOutputParser` in practice**
> ```python
> from pydantic import BaseModel, Field
> from langchain_core.output_parsers import PydanticOutputParser
>
> class Movie(BaseModel):
>     title: str = Field(description="Movie title")
>     year: int = Field(description="Release year")
>
> parser = PydanticOutputParser(pydantic_object=Movie)
> print(parser.get_format_instructions())  # inject this into your prompt
> ```
> **Tip:** the `format_instructions` string generated by the parser should be inserted directly into the prompt template — this is how the model "knows" the exact JSON shape it must return. If parsing still fails intermittently, `RetryOutputParser` / `OutputFixingParser` can re-prompt the model to correct malformed output.

---

# 6. Important Chain Execution Methods

### Module 5

Once prompts, models, runnables, and parsers are understood, learn how LangChain components are actually executed.

### Topics

* `.invoke()`
* `.batch()`
* `.stream()`
* `.ainvoke()`
* `.abatch()`
* `.astream()`
* `.astream_events()`
* Synchronous execution
* Asynchronous execution
* Batch execution
* Streaming execution
* Async batch processing
* Difference between `.stream()` and `.astream()`
* Parallel processing concepts

### Goal

Understand the different ways a LangChain chain can be executed depending on the application's requirements.

> 📘 **Deep Dive — Which method to use when**
>
> | Method | Sync/Async | Use case |
> |---|---|---|
> | `.invoke()` | Sync | Single input, single output — simplest case |
> | `.batch()` | Sync | Many independent inputs processed together |
> | `.stream()` | Sync | Token-by-token output for a single request (e.g. CLI apps) |
> | `.ainvoke()` | Async | Same as `.invoke()` inside an async web server |
> | `.abatch()` | Async | High-throughput batch processing without blocking the event loop |
> | `.astream()` | Async | Streaming inside async frameworks (FastAPI, etc.) — pairs naturally with Module 9 |
> | `.astream_events()` | Async | Fine-grained events (per-step, per-token) — useful for debugging agent chains |
>
> **Rule of thumb:** use the sync methods (`.invoke`, `.batch`, `.stream`) for scripts and notebooks; switch to the `a`-prefixed async methods once the chain is served behind a web framework like FastAPI, so a single slow LLM call doesn't block other requests.

---

# 7. LCEL — LangChain Expression Language

### Module 6

LCEL brings the previous concepts together and provides a clean way to compose LangChain components.

### Topics

* LangChain Expression Language
* LCEL syntax
* Runnable composition
* Pipe operator `|`
* Prompt → Model → Parser
* Building a chatbot with LCEL
* Single-session conversation
* Multi-session conversation
* Conversation history
* `ChatMessageHistory`
* `RunnableWithMessageHistory`

### Core Idea

```text
Input
  ↓
Prompt
  ↓
Model
  ↓
Parser
  ↓
Output
```

LCEL allows these components to be composed into a single pipeline.

### Goal

Understand how individual LangChain components become a complete application pipeline.

> 📘 **Deep Dive — The pipe operator, concretely**
> ```python
> from langchain_core.output_parsers import StrOutputParser
>
> chain = prompt | model | StrOutputParser()
> chain.invoke({"domain": "cooking", "question": "How do I poach an egg?"})
> ```
> Because `prompt`, `model`, and `StrOutputParser()` are all `Runnable`s (Module 3), the `|` operator simply wires the output of one into the input of the next — this is the same idea as Unix pipes, applied to LLM components. This is why LCEL chains automatically support `.batch()`, `.stream()`, and their async equivalents without any extra code.

---

# 8. Conversation Memory

### Module 6 & Module 7

After creating a chatbot, the next problem is conversation history.

A chatbot needs to remember previous messages so that future responses have the required context.

### Topics

* Chat history
* Conversation state
* `HumanMessage`
* `AIMessage`
* `ChatMessageHistory`
* Session-based conversations
* Multiple users / multiple sessions
* `RunnableWithMessageHistory`
* Storing conversation history
* Retrieving previous messages

### Goal

Understand how conversation state is maintained across multiple interactions.

> 📘 **Deep Dive — Wiring memory into an LCEL chain**
> ```python
> from langchain_community.chat_message_histories import ChatMessageHistory
> from langchain_core.runnables.history import RunnableWithMessageHistory
>
> store = {}
> def get_session_history(session_id: str):
>     if session_id not in store:
>         store[session_id] = ChatMessageHistory()
>     return store[session_id]
>
> chain_with_history = RunnableWithMessageHistory(chain, get_session_history)
> chain_with_history.invoke(
>     {"question": "What's the capital of France?"},
>     config={"configurable": {"session_id": "user-123"}},
> )
> ```
> **Key idea:** the `session_id` is what lets a single deployed chain serve *many independent conversations* at once — each session gets its own history object, keyed by ID, rather than one global history shared by all users.

---

# 9. Memory Optimization

### Module 7

Keeping the complete conversation forever can cause the context to grow continuously.

This module explores ways to control conversation history.

### Topics

* Why conversation history needs optimization
* Sliding-window memory
* Last `N` messages
* Manual history trimming
* `RunnableWithMessageHistory`
* Summarized conversation memory
* Comparing different memory strategies
* Context size considerations

### Main Approaches

```text
Full History
     ↓
Sliding Window
     ↓
Summarized History
```

### Goal

Understand how to maintain useful conversational context without unnecessarily sending the entire conversation every time.

> 📘 **Deep Dive — Trade-offs between strategies**
>
> | Strategy | Pros | Cons |
> |---|---|---|
> | Full history | Nothing is lost | Context grows unbounded, cost/latency increases over time |
> | Sliding window (last *N*) | Cheap, simple, bounded cost | Older but still-relevant context is silently dropped |
> | Summarized history | Keeps long-range context compactly | Summarization itself costs an extra LLM call and can lose nuance |
>
> **Practical guidance:** sliding-window memory is usually enough for short task-oriented chats (support bots, quick Q&A). Summarized memory is worth the extra complexity for long-running assistants where early context (e.g. a user's stated preferences) still matters many turns later.

---

# 10. Sequential & Conditional Chains

### Module 8

Not every application requires an agent.

Many applications can be solved using deterministic chains where the sequence of operations is already known.

### Topics

* Sequential chains
* Multiple-step workflows
* Intermediate outputs
* Final output
* Simple sequential chains
* `RunnableSequence`
* `.pipe()`
* `RunnablePassthrough`
* `RunnableLambda`
* `RunnableBranch`
* Conditional routing
* Routing based on input
* Combining multiple chains

### Example Concept

```text
Input
  ↓
Step 1
  ↓
Step 2
  ↓
Step 3
  ↓
Final Output
```

Conditional workflow:

```text
                 ┌── Chain A
Input → Router ──┤
                 └── Chain B
```

### Goal

Understand how to build predictable multi-step workflows before moving into agents.

> 📘 **Deep Dive — `RunnableBranch` for conditional routing**
> ```python
> from langchain_core.runnables import RunnableBranch
>
> branch = RunnableBranch(
>     (lambda x: "refund" in x["query"].lower(), refund_chain),
>     (lambda x: "technical" in x["query"].lower(), support_chain),
>     general_chain,  # default fallback
> )
> ```
> **When to reach for this instead of an agent:** if you already know the finite set of possible paths (e.g. "billing" vs "technical" vs "general"), a `RunnableBranch` is more predictable, cheaper, and easier to test than letting an LLM agent decide the route at runtime. Save agents (Module 13) for cases where the set of possible actions is open-ended or genuinely depends on reasoning.

---

# 11. Frontend & Streaming Integration

### Module 9

This module introduces how a LangChain chatbot can be connected to a frontend application.

### Topics

* FastAPI
* API endpoints
* Streaming chatbot responses
* Backend chatbot integration
* React frontend
* Frontend ↔ Backend communication
* Live response streaming

### Learning Focus

This module is primarily for understanding how a LangChain application can be connected to an actual user interface.

### Goal

Understand the connection between:

```text
React Frontend
      ↓
FastAPI Backend
      ↓
LangChain
      ↓
LLM
```

> 📘 **Deep Dive — Streaming endpoint sketch**
> ```python
> from fastapi import FastAPI
> from fastapi.responses import StreamingResponse
>
> app = FastAPI()
>
> @app.post("/chat")
> async def chat(payload: dict):
>     async def token_generator():
>         async for chunk in chain.astream(payload):
>             yield chunk
>     return StreamingResponse(token_generator(), media_type="text/plain")
> ```
> This is exactly where `.astream()` from Module 6 becomes practically useful: FastAPI's `StreamingResponse` forwards each token to the browser as soon as it's generated, which is what produces the familiar "typing" effect in chat UIs.

---

# 12. RAG — Retrieval-Augmented Generation

### Module 10

RAG introduces an important transition from simple LLM applications to applications that can work with **external knowledge**.

### Why RAG?

An LLM cannot automatically know the private or newly provided information contained in your own documents or databases.

RAG provides a workflow for retrieving relevant information and giving it to the model as context.

### Core RAG Flow

```text
Documents
    ↓
Load
    ↓
Split
    ↓
Create Embeddings
    ↓
Store in Vector Database
    ↓
Retrieve Relevant Chunks
    ↓
Provide Context to LLM
    ↓
Generate Answer
```

### Topics

* What is RAG?
* Why RAG is required
* Document loading
* Text splitting
* Embeddings
* Vector databases
* Similarity search
* Retrievers
* Making vector stores usable as Runnables
* `RunnablePassthrough`
* Context retrieval
* RAG chain
* LLM response generation

### Vector Store

The module also demonstrates how a vector store can be converted into a Retriever/Runnable so that it can participate naturally in a LangChain pipeline.

### Goal

Understand the complete conceptual flow of a basic RAG application.

> 📘 **Deep Dive — A minimal RAG chain**
> ```python
> retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
>
> rag_chain = (
>     {"context": retriever, "question": RunnablePassthrough()}
>     | prompt
>     | model
>     | StrOutputParser()
> )
> rag_chain.invoke("What does the report say about Q3 revenue?")
> ```
> **Common pitfall:** RAG quality is bottlenecked by *retrieval* quality, not model quality — if the wrong chunks are retrieved, the model will confidently answer from the wrong context. This is why Modules 13–14 (loaders, splitting) matter as much as the LLM call itself.

---

# 13. Document Loaders

### Module 11

Before building RAG systems, documents need to be loaded into LangChain.

### Topics

* Document loaders
* `Document`
* Text files
* `TextLoader`
* CSV files
* `CSVLoader`
* Web pages
* `WebBaseLoader`
* Lazy loading
* `lazy_load()`
* Unstructured document loading

### Important Idea

Different data sources require different loaders.

```text
Text File  → TextLoader
CSV        → CSVLoader
Web Page   → WebBaseLoader
Documents  → Unstructured Loaders
```

### Goal

Understand how external data is brought into LangChain before processing and retrieval.

> 📘 **Deep Dive — Why `lazy_load()` matters**
> `.load()` reads the entire source into memory as a list of `Document` objects immediately. `.lazy_load()` returns a generator that yields documents one at a time — for a folder with thousands of files or a huge CSV, this avoids loading everything into RAM at once.
> ```python
> for doc in loader.lazy_load():
>     process(doc)  # handle one Document at a time
> ```
> **Practical note:** every loader produces the same `Document` object (`page_content` + `metadata`), regardless of source type — this uniform shape is what lets the same splitting/embedding pipeline (Module 12) work no matter where the data came from.

---

# 14. Document Splitting

### Module 12

Large documents cannot always be embedded and retrieved as one large piece.

Document splitting breaks documents into meaningful chunks.

### Topics

* Why documents need to be split
* Chunk size
* Chunk overlap
* Token-based splitting
* Character-based splitting
* `CharacterTextSplitter`
* `TokenTextSplitter`
* `RecursiveCharacterTextSplitter`

### Recursive Character Splitting

Understand why `RecursiveCharacterTextSplitter` is commonly useful for general text because it attempts to preserve related text while creating smaller chunks.

### Document-Aware Splitting

Different document formats require different strategies.

#### Markdown

* `MarkdownHeaderTextSplitter`
* Splitting based on Markdown headers
* Combining header splitting with recursive splitting

#### JSON

* JSON-aware splitting
* `.split_json()`
* Handling nested key-value information

#### HTML

* `HTMLHeaderTextSplitter`
* Splitting based on HTML structure

#### Code

* `Language`
* Language-aware recursive splitting
* Splitting programming code according to language structure

### Goal

Understand that **chunking strategy directly affects the quality of retrieval** in RAG.

> 📘 **Deep Dive — Picking `chunk_size` and `chunk_overlap`**
> ```python
> from langchain_text_splitters import RecursiveCharacterTextSplitter
>
> splitter = RecursiveCharacterTextSplitter(
>     chunk_size=1000,     # characters (or tokens, if using a token-aware splitter)
>     chunk_overlap=150,   # ~15% overlap keeps context across chunk boundaries
> )
> chunks = splitter.split_documents(documents)
> ```
> **Starting guidance:**
> * Smaller chunks (300–500) → more precise retrieval, but risk losing surrounding context.
> * Larger chunks (1000–1500) → more context per chunk, but retrieval becomes less precise and can pull in irrelevant text.
> * `chunk_overlap` (typically 10–20% of `chunk_size`) prevents a sentence or idea from being cut in half exactly at a chunk boundary.
> There's no universally "correct" value — this is worth experimenting with against your own retrieval evaluation (see *Extra Learning → Retrieval Quality*).

---

# 15. Agents & Tools

### Module 13

Chains are useful when the workflow is known in advance.

Agents become useful when the model needs to decide **what action should happen next**.

### Topics

* Agents
* Tools
* Tool calling
* Agent workflow
* Tool descriptions
* Tool docstrings
* Tool arguments
* `args_schema`
* Pydantic schemas for tools
* `create_tool_calling_agent`
* `AgentExecutor`
* Structured chat agents
* `create_structured_chat_agent`

### Tool Calling

A tool gives the model access to an external function.

```text
User
 ↓
Agent
 ↓
Decide whether a tool is required
 ↓
Tool
 ↓
Tool Result
 ↓
Agent
 ↓
Final Answer
```

### Important Concept

The tool's **docstring is extremely important** because it describes the tool to the model.

For complex tools, `args_schema` can be used to describe and validate the expected arguments.

### Tool Calling vs Structured Chat

The module also compares:

* `create_tool_calling_agent`
* `create_structured_chat_agent`

The structured-chat approach is mainly useful when the model does not support native tool calling.

### Goal

Understand how an LLM changes from simply generating answers to **selecting and using external tools**.

> 📘 **Deep Dive — Defining a tool with `@tool`**
> ```python
> from langchain_core.tools import tool
>
> @tool
> def get_weather(city: str) -> str:
>     """Get the current weather for a given city name."""
>     return f"It is sunny in {city}."
> ```
> The docstring (`"""Get the current weather..."""`) is not just documentation for humans — it's sent to the model as part of the tool's definition, and it's the *primary signal* the model uses to decide when to call this tool. A vague docstring like `"""Weather tool."""` leads to unreliable tool selection; a specific one describing exactly what the tool does and what input it expects leads to much more consistent agent behavior.

---

# 16. Advanced Agentic RAG

### Module 14

This module combines the concepts learned throughout the repository.

Instead of using RAG as a fixed pipeline, the retrieval system can become a **tool available to an agent**.

### Learning Flow

```text
Documents
    ↓
Document Loader
    ↓
Document Splitting
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retriever
    ↓
RAG
    ↓
Convert RAG into a Tool
    ↓
Agent
    ↓
Agent decides when to use the Tool
    ↓
Retrieve Information
    ↓
Generate Final Answer
```

### Topics

* Advanced agent flow
* Vector stores
* Chroma
* Retriever
* RAG
* Runnable RAG
* Converting RAG into a tool
* Tool descriptions
* Agent + RAG
* Multiple tools
* Tool execution
* Agent decision making
* AgentExecutor
* SQL agents
* Database tools
* Web search tools
* SQLDatabaseToolkit

### Core Concept

The important transition is:

```text
Traditional RAG

Question
   ↓
Retrieve
   ↓
Generate
```

to:

```text
Agentic RAG

Question
   ↓
Agent
   ↓
Should I retrieve?
   ↓
Which tool should I use?
   ↓
Retrieve / Search / Database
   ↓
Observe result
   ↓
Continue or answer
```

### Goal

Understand how RAG, tools, and agents can be combined to create more flexible LLM applications.

> 📘 **Deep Dive — Turning a retriever into a tool**
> ```python
> from langchain.tools.retriever import create_retriever_tool
>
> retriever_tool = create_retriever_tool(
>     retriever,
>     name="search_company_docs",
>     description="Search internal company documents for policy and product information.",
> )
> tools = [retriever_tool, get_weather, sql_tool]
> agent_executor = AgentExecutor(agent=agent, tools=tools)
> ```
> **The core insight:** in fixed RAG (Module 10), retrieval happens on *every* query whether it's needed or not. In agentic RAG, the agent first reasons about whether retrieval is even necessary, and if there are multiple tools (retriever, SQL, web search), it chooses the *right* one for the question — this is the ReAct-style "reason → act → observe → repeat" loop.

---

# Extra Learning

The following topics appear in the notebooks or naturally extend the concepts being practiced, but they are better treated as **additional learning** rather than part of the main progression.

## 1. SQL / Database Agents

Explore how agents can interact with databases using:

* `create_sql_agent`
* `SQLDatabase`
* `SQLDatabaseToolkit`
* Database tools
* Natural language → SQL
* SQL execution through agents

Learning flow:

```text
User Question
      ↓
Agent
      ↓
Understand Question
      ↓
Generate SQL
      ↓
Database
      ↓
Result
      ↓
Agent
      ↓
Answer
```

> 📘 **Deep Dive:** Always run SQL agents against a **read-only** database connection or user role while learning. An agent that can generate arbitrary SQL can also generate destructive statements (`DROP`, `DELETE`) if the question is ambiguous or the prompt is manipulated — this is a well-known risk with natural-language-to-SQL systems.

---

## 2. Web Search Tools

Learn how external web-search functionality can be exposed to an agent as a tool.

Topics:

* Search tools
* Tool descriptions
* Agent-selected search
* Search results as tool output
* Agent + web search

---

## 3. Agentic RAG

Go deeper into the difference between:

* Fixed RAG
* RAG as a tool
* Agentic RAG
* Multiple retrieval steps
* Tool selection
* Retrieval decisions

---

## 4. Streaming

Go deeper into:

* `.stream()`
* `.astream()`
* `.astream_events()`
* Token streaming
* Event streaming
* Streaming agent responses
* Streaming through APIs
* Frontend live updates

---

## 5. Async LangChain

Explore:

* `async`
* `await`
* `.ainvoke()`
* `.abatch()`
* `.astream()`
* `asyncio.gather()`
* Concurrent model calls

> 📘 **Deep Dive:** `asyncio.gather()` combined with `.ainvoke()` is how you fire off several independent LLM calls concurrently instead of sequentially — e.g. summarizing 10 documents at once instead of one after another. This is different from `.abatch()`, which batches a *single* chain over multiple inputs; `asyncio.gather()` is more general and can run entirely different chains/tasks concurrently.

---

## 6. Retrieval Quality

After understanding basic RAG, continue with:

* Similarity search
* `k`
* Metadata filtering
* MMR
* Hybrid search
* Retrieval evaluation
* Chunk-size experiments
* Chunk-overlap experiments
* Embedding model comparison

> 📘 **Deep Dive:** MMR (Maximal Marginal Relevance) is worth learning early here — plain similarity search can return several near-duplicate chunks that all say the same thing, while MMR explicitly balances relevance against diversity so the retrieved context covers more distinct information for the same `k`.

---

## 7. Embeddings

Study embeddings more deeply:

```text
Text
 ↓
Embedding Model
 ↓
Vector
 ↓
Vector Database
 ↓
Similarity Search
```

Important areas:

* Embedding dimensions
* Cosine similarity
* Semantic similarity
* Query embeddings
* Document embeddings
* Embedding model selection

---

## 8. Vector Databases

After learning the basic RAG implementation, explore different vector stores and compare their approaches.

Topics:

* FAISS
* Chroma
* Pinecone
* Weaviate
* Similarity search
* Metadata filtering
* Persistence
* Retrieval configuration

> 📘 **Deep Dive:** FAISS and Chroma are the easiest to start with because they run locally with no external account — good for learning. Pinecone and Weaviate are managed/hosted services better suited once you need production-scale persistence, multi-user access, or horizontal scaling beyond what fits comfortably on one machine.

---

# Recommended Learning Sequence

For the best understanding, follow the notebooks in this order:

```text
01  LangChain Basics
        ↓
02  Prompt Templates
        ↓
03  Model Parameters
        ↓
04  Runnables
        ↓
05  Output Parsers
        ↓
06  Chain Execution Methods
        ↓
07  LCEL
        ↓
08  Conversation Memory
        ↓
09  Memory Optimization
        ↓
10  Sequential & Conditional Chains
        ↓
11  Frontend & Streaming
        ↓
12  RAG
        ↓
13  Document Loaders
        ↓
14  Document Splitting
        ↓
15  Agents & Tools
        ↓
16  Advanced Agentic RAG
        ↓
17  Extra Learning
```

---

# The Big Picture

The concepts in this repository can ultimately be connected into one larger mental model:

```text
                         LANGCHAIN
                             │
             ┌───────────────┼───────────────┐
             │               │               │
           Models          Prompts         Tools
             │               │               │
             └───────────────┼───────────────┘
                             │
                         Runnables
                             │
                            LCEL
                             │
                    ┌────────┴────────┐
                    │                 │
                  Chains            Agents
                    │                 │
                    │                 ├── Tools
                    │                 ├── RAG
                    │                 ├── Web Search
                    │                 └── Databases
                    │
             Conversation Memory
                    │
                    └──────────────┐
                                   │
                                  RAG
                                   │
                    ┌──────────────┼──────────────┐
                    │              │              │
                Documents      Splitting      Embeddings
                    │              │              │
                    └──────────────┼──────────────┘
                                   │
                             Vector Store
                                   │
                               Retriever
                                   │
                                  RAG
                                   │
                                Agent
                                   │
                          Agentic Applications
```

---

# Final Learning Goal

By completing this learning path, the main progression should be clear:

```text
LLM
 ↓
Prompt
 ↓
Runnable
 ↓
Chain
 ↓
LCEL
 ↓
Memory
 ↓
RAG
 ↓
Retriever
 ↓
Tool
 ↓
Agent
 ↓
Agentic RAG
```

The most important thing is not memorizing individual LangChain classes.

The real objective is to understand **how information flows through an LLM application**, how each abstraction solves a specific problem, and how those abstractions can be combined to build increasingly capable systems.

---

> ℹ️ **Note on this document:** Sections marked with 📘 **Deep Dive** are supplementary learning notes (code snippets, comparison tables, trade-offs, and practical tips) added to reinforce the concepts already outlined above. They do not replace or alter any of the original module descriptions — all original content, structure, and headings remain exactly as originally written.