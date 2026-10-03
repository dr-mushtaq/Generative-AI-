A stage-by-stage guide to going from zero to building real GenAI applications, with one running example to tie every concept together.
Develop with Devshree
Jul 25, 2026

Generative AI is no longer a specialisation. In 2026, it is a baseline skill, the way knowing DSA was for SWE roles five years ago. Companies are not experimenting with it anymore. They are hiring for it.

But most roadmaps just dump a list of 50 tools and leave you guessing how they fit together.

This one is different. Every concept is explained through a single project you will build from scratch:

SmartApply, an AI-powered job application assistant that reads a job description, matches it to your resume, writes a tailored cover letter, and tracks all your applications in one place.

By the end of this roadmap, you will know exactly how to build it, and you will have touched every major layer of the GenAI stack.

If you like my work and find my resources useful, consider subscribing to this Newsletter to get them in your mailbox weekly :)

Type your email...
Subscribe
Stage 1: Python and ML Fundamentals
Before touching any AI model, you need a solid base. Skip this and everything else will feel like magic you cannot control.

# ✅ Python
Core Python: Data types, loops, functions, list comprehensions, error handling

Object-Oriented Programming: Classes, inheritance. You will use this when structuring your AI pipelines

File handling: Reading PDFs, CSVs, and text files (critical for processing resumes and job descriptions)

Libraries to learn:

NumPy and Pandas for data manipulation

Matplotlib and Seaborn for visualising model outputs

PyPDF2 or pdfplumber for extracting text from PDFs

# ✅ Machine Learning Basics
You do not need to be an ML researcher. But you do need to understand:

How models learn from data (supervised vs. unsupervised learning)

What a loss function is: the measure of how wrong the model’s prediction is

Gradient descent: the algorithm that reduces error by adjusting model parameters step by step

Train/test splits, overfitting, and evaluation metrics

✅ Tools
IDE: VS Code with the Python extension

Environment: Jupyter Notebooks for experimentation

Version control: Git and GitHub. Push every project from day one

Back to SmartApply: You write a Python script that uses pdfplumber to extract text from your resume PDF, cleans it with Pandas, and saves it as structured data. This becomes the input to every AI model later.

Resources:

[📺 Learn Python Full Course for Beginners (freeCodeCamp, 50M+ views)](https://youtu.be/rfscVS0vtbw?si=OioFxFP0kUiX9_C7)

[📺 Python Full Course for free, 2024 (Bro Code)](https://www.youtube.com/watch?v=ix9cRaBkVe0)

# Stage 2: Understanding LLMs

What is an LLM? A Large Language Model is a neural network trained on billions of text examples to predict the next word in a sequence. Because of the scale of training, it develops an understanding of language, reasoning, and even code. GPT-4, Claude, and Gemini are all LLMs.

You do not need to train one. You need to understand how they work well enough to use them correctly.

✅ Core Concepts to Learn
Transformer architecture: The “attention mechanism” is the core idea. The model learns which words in a sentence are most relevant to each other. Watch Andrej Karpathy’s videos below for the clearest intuition

Tokens: LLMs do not read words. They read tokens, chunks of text (roughly 4 characters each). “ChatGPT” is 3 tokens. This matters because every API call is billed by tokens, and models have a token limit

Context window: The maximum amount of text the model can “see” at once. GPT-4o has a 128K token context window. If your input exceeds it, the model forgets earlier content

Temperature: Controls randomness in output. Temperature 0 = deterministic (same answer every time). Temperature 1 = more creative and varied

Inference APIs: You do not need a GPU to use LLMs. Call them via API

✅ APIs to Get Familiar With
OpenAI API (GPT-4o, GPT-4o mini)

Anthropic Claude API (Claude 3.5 Sonnet, Claude Haiku)

Google Gemini API (Gemini 1.5 Pro)

Open-source models via Ollama: Run LLaMA 3, Mistral, Phi-3 locally for free

Back to SmartApply: You send the cleaned resume text and a job description to the OpenAI API and ask it to return a match score and the top 3 overlapping skills. You have just built the core of your application with 20 lines of Python.

Resources:

[📺 Intro to Large Language Models (Andrej Karpathy, 1hr talk)](https://youtu.be/zjkBMFhNj_g?si=I46qP9d6lm7rBETT)

[📺 Let’s build GPT: from scratch, in code (Andrej Karpathy)](https://youtu.be/kCc8FmEb1nY?si=r1eBsKN-_yB7-Cnq)

# Stage 3: Prompt Engineering

What is it? Prompt engineering is the skill of writing instructions that get the best possible output from an LLM. The model is the same. What changes is how you talk to it.

A bad prompt gives you a generic, vague answer. A well-crafted prompt gives you a structured, accurate, task-specific response.

✅ Techniques to Master
Zero-shot prompting: Giving the model a task with no examples. Works for simple tasks

Few-shot prompting: Providing 2 to 3 examples inside the prompt so the model learns the pattern from context

Chain-of-thought prompting: Asking the model to “think step by step” before answering. Dramatically improves accuracy on reasoning tasks

Role prompting: Assigning the model a persona (”You are a senior recruiter with 10 years of experience at Google...”)

Output formatting: Asking the model to return JSON, tables, or numbered lists so you can parse the response programmatically

System prompt vs. user prompt: The system prompt sets behaviour and constraints. The user prompt is the actual query

✅ Common Mistakes to Avoid
Being too vague. “Write a cover letter” vs. “Write a 3-paragraph cover letter that matches the top 3 skills from the job description to the candidate’s resume. Use a professional tone. Avoid phrases like ‘I am excited to’ or ‘I believe I would be a great fit’”

Not specifying output format when you need to parse the response in code

Ignoring the system prompt. It is your most powerful tool for consistent behaviour

Back to SmartApply: Instead of prompting “Write a cover letter”, you use:

System: You are a career coach who has helped 500+ candidates land roles at top tech companies. Write concise, specific cover letters that highlight measurable impact.

User: Given the job description and resume below, write a 3-paragraph cover letter. Paragraph 1: connect the candidate’s strongest matching skill to the role. Paragraph 2: highlight one project with a quantified result. Paragraph 3: closing with availability. Return as plain text.

The output is dramatically better.

Resources:

[📺 Prompt Engineering Tutorial – Master ChatGPT and LLM Responses (freeCodeCamp)](https://youtu.be/_ZvnD73m40o?si=OLNPgdg8idkZtQ3B)

[📺 Prompt Engineering Full Course: From Beginner to Pro](https://youtu.be/YhRfgYH_AoU?si=LNwbIQMZLz4jIE0q)

# Stage 4: RAG (Retrieval-Augmented Generation)
What is it? LLMs have a knowledge cutoff. They do not know about data you collected today, documents you wrote last week, or the 300 job listings you bookmarked. RAG fixes this by giving the model access to a searchable database before it generates a response.

Think of it as: look it up first, then answer.

✅ Core Concepts
Embeddings: Text converted into a list of numbers (a vector) that captures its meaning. Two sentences with similar meaning will have similar vectors, even if the words are different. This is what makes semantic search possible

Vector databases: Databases that store embeddings and let you search by meaning, not just keywords

Start with ChromaDB (local, easy to set up)

Production options: Pinecone, Weaviate, Qdrant

Retrieval step: When a query comes in, convert it to an embedding, find the most similar embeddings in the database, and return those chunks of text

Augmentation step: Pass the retrieved text as context to the LLM, then ask it to answer based only on that context

Chunking strategy: How you split documents before storing matters. Too large, you waste context. Too small, you lose meaning. 500 to 1000 tokens per chunk with overlap is a good starting point

✅ Frameworks
LangChain: Most popular. Has built-in document loaders, text splitters, retrievers, and chains

LlamaIndex: Purpose-built for RAG, often simpler for pure retrieval use cases

LangGraph: For more complex, stateful RAG pipelines (covered in Stage 5)

Back to SmartApply: You have bookmarked 300 job listings. You cannot paste all of them into every prompt. It would exceed the context window and cost a fortune. Instead:

You chunk each job description into 500-token segments

Convert each chunk into an embedding using text-embedding-3-small (OpenAI)

Store all embeddings in ChromaDB

When you ask “Which jobs match my Python and ML skills?”, your query is embedded and the 5 most semantically similar job chunks are retrieved

Those 5 chunks are passed to the LLM, which ranks them and explains why each is a good fit

That is RAG. The model now “knows” about your 300 jobs without you ever exceeding the context window.

Resources:

[📺 RAG Explained in 20 Minutes, with Hands-on Project](https://youtu.be/RosLeHGBLoY?si=uNMlxc_-UJugrdnW)

[📺 Complete RAG Tutorial 2025: Build AI Apps with Retrieval Augmented Generation](https://www.youtube.com/playlist?list=PLNIQLFWpQMRUMjxfe8o6g3uzJ6LH_VotY)

# Stage 5: AI Agents

What is it? An AI agent is an LLM that does not just answer questions. It takes actions. It can browse the web, run code, call APIs, write files, and loop through a series of steps until a goal is complete.

A standard LLM responds. An agent executes.

✅ Core Concepts
Tool use / Function calling: You define functions (tools) and tell the LLM what each one does. The model decides when to call them and with what arguments. For example: search_jobs(query: str), send_email(to: str, body: str), update_tracker(job_id: str, status: str)

ReAct pattern: The most common agent loop. Reason, Act, Observe, repeat. The model reasons about what to do, calls a tool, observes the result, then reasons again until done

Memory: Agents need to remember what they have done. Short-term memory lives in the conversation context. Long-term memory is stored externally in a vector DB or a simple database

Multi-agent systems: Instead of one agent doing everything, you build a team. A Researcher agent, a Writer agent, a Reviewer agent, each specialised and coordinated by an orchestrator

✅ Frameworks
LangGraph: Best for building stateful, multi-step agent workflows with clear control flow. Think of it as a graph where each node is an LLM call or tool call

CrewAI: Easier to set up for multi-agent pipelines. Define roles, goals, and backstories for each agent

AutoGen (Microsoft): Strong for multi-agent conversations and code-executing agents

Back to SmartApply: You turn SmartApply into a full agent. When you say “Find and apply to 10 ML engineer roles in San Francisco today”, it:

Calls a search_jobs tool to query LinkedIn and Indeed APIs

Filters results by matching them against your resume using RAG (Stage 4)

For each match, calls a generate_cover_letter tool (Stage 3)

Calls an update_tracker tool to log the application in your spreadsheet

Drafts emails using a draft_email tool and surfaces them for your approval before sending

That is a multi-step agent completing a real-world task autonomously.

Resources:

[📺 LangGraph Tutorial: How to Build Advanced AI Agent Systems](https://youtu.be/1w5cCXlh7JQ?si=_zC0psZ4jt5Q2MMB)

[📺 AI Agent Tutorial for Beginners (LangGraph Playlist)](https://www.youtube.com/playlist?list=PLCqo9x4xS5_o5ns-kYB1SN-UsI6GCV5qX)

# Stage 6: Fine-Tuning
What is it? Fine-tuning means taking a pre-trained LLM and training it further on your specific dataset so it internalises your domain, tone, or output format.

Important: most applications do not need fine-tuning. Prompt engineering (Stage 3) and RAG (Stage 4) solve 80 to 90% of problems. Fine-tune only when:

You need highly consistent output format across thousands of calls

Your domain has very specific terminology that base models handle poorly

You have 500+ high-quality labelled input/output pairs

Latency matters and you want a smaller, faster model

✅ Core Concepts
Full fine-tuning vs. PEFT: Full fine-tuning updates all model weights (expensive, needs significant GPU). PEFT (Parameter-Efficient Fine-Tuning) updates only a small fraction of weights

LoRA (Low-Rank Adaptation): The most popular PEFT method. Instead of updating all parameters, it trains small adapter matrices and merges them in. Achieves near full fine-tune performance at a fraction of the cost

Dataset preparation: Your data needs to be in instruction format: {"instruction": "...", "input": "...", "output": "..."}. Quality matters far more than quantity

Evaluation: Use BLEU and ROUGE scores for text quality, but always complement with human evaluation. Automated metrics miss nuance

✅ Tools
Hugging Face Transformers + PEFT library: The standard stack for fine-tuning open-source models

Unsloth: Makes LoRA fine-tuning 2x faster and uses 60% less memory. Great for free-tier GPUs

Google Colab / Kaggle Notebooks: Free GPU access for small fine-tuning experiments

Weights and Biases (W&B): For tracking your training runs, loss curves, and experiments

Back to SmartApply: You collect 400 cover letters that resulted in interviews (from your own history and publicly shared examples). You fine-tune LLaMA 3.1 8B using LoRA on this dataset. The model now writes in a style that is not just grammatically correct, but pattern-matched to letters that actually convert.

Resources:

📺 Generative AI Fine Tuning LLM Models Crash Course (Krish Naik)

📺 Fine-tuning LLM Full Playlist (Krish Naik)

Stage 7: Deployment and LLMOps
What is it? Building something in a Jupyter Notebook is step one. Getting it to run reliably for real users, at scale, with monitoring, is the actual engineering challenge.

LLMOps (LLM Operations) covers deploying, monitoring, and maintaining LLM-powered applications in production.

✅ Core Concepts
API wrapping: Use FastAPI to expose your application as a REST API so a frontend or other services can call it

Containerisation with Docker: Package your app and all its dependencies into a container so it runs identically on any machine. One command to run, one command to deploy

Cloud deployment:

AWS: Lambda (serverless), SageMaker (model hosting), EC2 (full control)

GCP: Cloud Run, Vertex AI

Azure: Azure AI Studio, Container Apps

Monitoring: Log every prompt and response. Track latency (how slow is it?), cost (how many tokens per call?), and failure rate (how often does it return garbage?)

Guardrails: Safety layers that check output before returning it to the user. Prevents the model from hallucinating, going off-topic, or returning harmful content. Libraries: Guardrails AI, NeMo Guardrails

Cost management: LLM API calls add up fast. Cache frequent responses, use smaller models (GPT-4o mini instead of GPT-4o) where full capability is not needed, and set token limits per call

✅ Tooling
FastAPI: Lightweight, fast, industry standard for Python APIs

Docker + Docker Compose: For containerising your full stack

LangSmith: LangChain’s observability tool. Traces every LLM call, shows you exactly what prompt went in and what came out

GitHub Actions: CI/CD to automate testing and deployment on every code push

Back to SmartApply: You wrap SmartApply in a FastAPI app, containerise it with Docker, and deploy it on AWS Lambda. Every cover letter generation call is logged in LangSmith. You set up an alert if latency exceeds 5 seconds or token cost per call exceeds $0.02. You add a Guardrails check to ensure the output is always professional in tone. SmartApply is now a real product.

Resources:

📺 FastAPI + LangServe: The Secret to Deploying Your LLM App

📺 How to Deploy ML Solutions with FastAPI, Docker, and AWS

Where to Go From Here
If you build SmartApply across these 7 stages, you will have touched APIs, RAG, agents, fine-tuning, and production deployment, all in one project. That is not just a tutorial project. That is a portfolio piece you can demo, a GitHub repo you can show, and a real problem you solved for yourself.

GenAI is moving fast, but the fundamentals, transformers, embeddings, agents, RAG, are not going anywhere. The specific tools will change every few months but the concepts underneath them will not.

One last thing: build in public. Write about what you are building on LinkedIn. The people who document their learning consistently outperform those who only code.

Final Thoughts
Priya started with a blank Python file and ended with a production-grade ML system. She didn’t get there by memorising algorithms. She got there by building things, shipping them, watching them fail, and fixing them. That’s what separates an ML Engineer from someone who’s done a few Kaggle tutorials.

If you want more resources on tech, AI, productivity and interview prep, follow me on:
