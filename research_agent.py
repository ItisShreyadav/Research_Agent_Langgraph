from typing import Dict, TypedDict, List, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json
from langgraph.graph import StateGraph

# --- LLM Setup ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
   # openai_api_key="sk-or-v1-",  # put your key here
    openai_api_base="https://openrouter.ai/api/v1"
)

class WorkflowState(TypedDict):
    messages: List[BaseMessage]
    workflow_params: Dict[str, Any]
    current_step: str
    is_complete: bool
    needs_user_input: bool
    input_request: str
    pending_workflow_step: Optional[str]
    total_tokens: int

# --- SIMULATED TOOLS ---

def validate_research_topic(topic: str) -> Dict[str, Any]:
    """
    Validates if a research topic is clear and searchable using LLM.
    """
    print(f"\n[Validating Topic]: {topic}")
    
    if not topic or not topic.strip():
        return {"is_valid": False, "reason": "Topic is empty"}
    
    validation_prompt = f"""
You are a research topic validator.

USER'S TOPIC: "{topic}"

TASK:
- Determine if this is a valid, searchable research topic
- A valid topic should be specific enough to research but not too narrow
- It should be clear what the user wants to know
- Return ONLY JSON, no explanation

RETURN FORMAT (JSON ONLY):
{{
    "is_valid": true,
    "reason": "Topic is clear and researchable"
}}
OR
{{
    "is_valid": false,
    "reason": "brief explanation of why it's invalid"
}}

RETURN JSON ONLY. NOTHING ELSE. NO EXPLANATIONS.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=validation_prompt)])
        content = response.content.strip().strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        return {
            "is_valid": False,
            "reason": f"Validation error: {str(e)}"
        }


def search_sources(topic: str) -> Dict[str, Any]:
    """
    PURE LLM TOOL.
    Generates research sources for a given topic.
    LLM decides what sources are relevant.
    No logic, no parsing, no fallback — raw LLM response (structured).
    """
    print(f"\n[Searching Sources]: {topic}")
    
    search_prompt = f"""
You are a research assistant that finds credible sources.

RESEARCH TOPIC: "{topic}"

TASK:
- Generate 5 realistic, credible sources for this research topic
- Include academic papers, reputable websites, news articles, or reports
- Each source should have: title, url (realistic format), summary (2-3 sentences)
- Make sources relevant and diverse
- If you cannot generate appropriate sources for this topic, return success=false with error

RETURN FORMAT (JSON ONLY):
{{
    "success": true,
    "sources": [
        {{
            "title": "...",
            "url": "https://...",
            "summary": "brief 2-3 sentence summary"
        }}
    ],
    "error": null
}}
OR
{{
    "success": false,
    "sources": [],
    "error": "reason for failure"
}}

RETURN JSON ONLY. NOTHING ELSE. NO EXPLANATIONS.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=search_prompt)])
        content = response.content.strip().strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        return {
            "success": False,
            "sources": [],
            "error": f"Search tool failed: {str(e)}"
        }


def summarize_research(sources: List[Dict[str, Any]], topic: str) -> Dict[str, Any]:
    """
    Summarizes research findings from sources using LLM.
    Creates a coherent summary of key findings.
    """
    print(f"\n[Summarizing Research]")
    
    if not sources:
        return {
            "success": False,
            "summary": "",
            "error": "No sources to summarize"
        }
    
    sources_text = "\n\n".join([
        f"SOURCE {i+1}:\nTitle: {s['title']}\nURL: {s['url']}\nSummary: {s['summary']}"
        for i, s in enumerate(sources)
    ])
    
    summary_prompt = f"""
You are a research analyst creating a summary.

RESEARCH TOPIC: "{topic}"

SOURCES:
{sources_text}

TASK:
- Create a comprehensive summary (4-6 paragraphs) that synthesizes these sources
- Identify key themes, findings, and insights
- Note any patterns or contradictions
- Write clearly and professionally
- Return ONLY JSON, no explanation

RETURN FORMAT (JSON ONLY):
{{
    "success": true,
    "summary": "your comprehensive summary here (4-6 paragraphs)...",
    "key_points": ["point 1", "point 2", "point 3"],
    "error": null
}}
OR
{{
    "success": false,
    "summary": "",
    "key_points": [],
    "error": "reason for failure"
}}

RETURN JSON ONLY. NOTHING ELSE. NO EXPLANATIONS.
"""
    
    try:
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        content = response.content.strip().strip("`").strip()
        if content.startswith("json"):
            content = content[4:].strip()
        
        parsed = json.loads(content)
        return parsed
    except Exception as e:
        return {
            "success": False,
            "summary": "",
            "key_points": [],
            "error": f"Summary failed: {str(e)}"
        }


def generate_final_answer(topic: str, summary: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates the final research answer/report using LLM.
    """
    print(f"\n[Generating Final Answer]")
    
    if not summary:
        return {
            "success": False,
            "answer": "",
            "error": "No summary to work with"
        }
    
    sources_list = "\n".join([
        f"{i+1}. {s['title']} - {s['url']}"
        for i, s in enumerate(sources)
    ])
    
    answer_prompt = f"""
You are a research report writer creating a final answer.

RESEARCH TOPIC: "{topic}"

SUMMARY:
{summary}

SOURCES:
{sources_list}

TASK:
- Create a well-structured research report
- Include sections: Overview, Key Findings, Analysis, Conclusion
- Write professionally and clearly
- Include source references
- Format with clear sections
- Return ONLY the report text, not JSON

Write the complete research report:
"""
    
    try:
        response = llm.invoke([HumanMessage(content=answer_prompt)])
        report_text = response.content.strip()
        
        return {
            "success": True,
            "answer": report_text
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "",
            "error": f"Answer generation failed: {str(e)}"
        }


def generate_contextual_message(context: str) -> str:
    """
    Uses LLM to generate a natural, user-friendly message based on context.
    """
    prompt = f"""
You are a friendly, helpful research assistant.

CONTEXT:
{context}

TASK:
- Generate a natural, conversational message to show to the user.
- Keep it short, polite, and clear.
- Do NOT use markdown, backticks, or JSON. Just plain text.
- Do NOT explain your reasoning — only return the message.

EXAMPLE:
Context: "User has not provided a research topic yet."
Message: "Hi! What topic would you like me to research for you?"

RETURN MESSAGE ONLY:
"""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().strip("`").strip()
    except Exception as e:
        return "I'm ready to help with your research. What would you like to know about?"


# --- NODE EXECUTION LOGIC ---

def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles user interaction and extracts research topic.
    """
    if state["needs_user_input"]:
        print(f"\n[Assistant]: {state['input_request']}")
        user_input = input("[User]: ")
        state["messages"].append(HumanMessage(content=user_input))

        extract_prompt = f"""
You are a research topic extractor.

TASK:
- Read the user input and extract ONLY the research topic as a string.
- If you can extract a clear topic, return: {{"research_topic": "extracted topic"}}
- If you CANNOT extract any topic, return: {{"research_topic": null}}
- Return RAW JSON ONLY — NO MARKDOWN, NO BACKTICKS, NO EXPLANATIONS, NO EXTRA FIELDS.

EXAMPLE 1:
User said: "I want to research climate change impacts"
RETURN:
{{"research_topic": "climate change impacts"}}

EXAMPLE 2:
User said: "hello there"
RETURN:
{{"research_topic": null}}

User said: "{user_input}"

RETURN JSON ONLY:
"""

        try:
            response = llm.invoke([HumanMessage(content=extract_prompt)])
            print("\nLLM response:", response)
            state["total_tokens"] += response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

            content = response.content.strip().strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
            
            parsed = json.loads(content)

            research_topic = parsed.get("research_topic")
            print("\nResearch topic:", research_topic)
            
            if isinstance(research_topic, str) and research_topic.strip():
                # ✅ Success — update workflow_params
                state["workflow_params"]["research_topic"] = research_topic.strip()
                state["messages"].append(AIMessage(content="Got it! Let me validate your research topic..."))
                state["needs_user_input"] = False
                state["input_request"] = ""
                state["pending_workflow_step"] = ""  # Clear it
            else:
                # ❌ research_topic is null, not string, or empty
                raise ValueError("Research topic is null or invalid")

        except Exception as e:
            print(f"[System]: LLM parsing failed: {e}")
            fallback_msg = generate_contextual_message(
                "The user's input did not contain a recognizable research topic. Ask them to provide a clear topic."
            )
            state["messages"].append(AIMessage(content=fallback_msg))
            state["input_request"] = fallback_msg
            state["pending_workflow_step"] = "chat_node"  # STAY HERE
            # → Do NOT update workflow_params
            # → Do NOT set needs_user_input=False → user retries

    return state


def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates the research topic.
    """
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["research_topic"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to provide a research topic.
        Keep it to one sentence. Avoid technical jargon.
        Example: "What topic would you like me to research?"
        """
        try:
            request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            request = "Please tell me what topic you'd like to research."
            
        return {
            **state,  # 👈 Preserve everything
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "validation_node",
            "current_step": "validation_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = validate_research_topic(params["research_topic"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result.get("is_valid"):
        error_detail = result.get("reason", "Topic is not clear enough")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User provided research topic: '{params["research_topic"]}'
        Respond in one clear sentence. Offer solution or alternative.
        """
        try:
            user_request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            user_request = f"The topic '{params['research_topic']}' needs to be clearer. Could you provide more specific details?"

        return {
            **state,  # 👈 Preserve everything
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "validation_node",
            "current_step": "validation_node"
        }

    # 4. Success — update state
    return {
        **state,  # 👈 Preserve everything
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "validation_result": result,
            "validated_topic": params["research_topic"]
        },
        "current_step": "validation_node",
        "messages": [
            *state.get("messages", []),
            AIMessage(content="✓ Topic validated! Searching for sources...")
        ]
    }


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Searches for research sources.
    """
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["research_topic"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to provide a research topic.
        Keep it to one sentence. Avoid technical jargon.
        Example: "I need a topic to search for — what would you like to research?"
        """
        try:
            request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            request = "Please provide a research topic to search for."
            
        return {
            **state,
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "search_node",
            "current_step": "search_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = search_sources(params["research_topic"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result.get("success"):
        error_detail = result.get("error", "Could not find sources")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User's research topic: '{params["research_topic"]}'
        Respond in one clear sentence. Offer solution or alternative.
        """
        try:
            user_request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            user_request = f"I couldn't find sources for '{params['research_topic']}'. Could you rephrase or try a different topic?"

        return {
            **state,
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "search_node",
            "current_step": "search_node"
        }

    # 4. Success — update state
    num_sources = len(result.get("sources", []))
    return {
        **state,
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "search_result": result,
            "sources": result.get("sources", [])
        },
        "current_step": "search_node",
        "messages": [
            *state.get("messages", []),
            AIMessage(content=f"✓ Found {num_sources} sources! Summarizing research...")
        ]
    }


def summarize_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Summarizes research findings from sources.
    """
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["sources", "research_topic"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to ensure sources were found first.
        Keep it to one sentence. Avoid technical jargon.
        Example: "I need sources before I can summarize — let me search first."
        """
        try:
            request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            request = "Please let me search for sources before summarizing."
            
        return {
            **state,
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "summarize_node",
            "current_step": "summarize_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = summarize_research(params["sources"], params["research_topic"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result.get("success"):
        error_detail = result.get("error", "Could not create summary")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User's research topic: '{params["research_topic"]}'
        Respond in one clear sentence. Ask for clarification or offer to try again.
        """
        try:
            user_request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            user_request = f"I had trouble summarizing the research on '{params['research_topic']}'. Could you clarify what you're looking for?"

        return {
            **state,
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "summarize_node",
            "current_step": "summarize_node"
        }

    # 4. Success — update state
    return {
        **state,
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "summary_result": result,
            "summary": result.get("summary", ""),
            "key_points": result.get("key_points", [])
        },
        "current_step": "summarize_node",
        "messages": [
            *state.get("messages", []),
            AIMessage(content="✓ Research summarized! Generating final answer...")
        ]
    }


def answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates the final research answer/report.
    Gives final results with a tool call.
    """
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["summary", "sources", "research_topic"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to ensure summary was created first.
        Keep it to one sentence. Avoid technical jargon.
        Example: "I need to complete the summary before generating the final answer."
        """
        try:
            request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            request = "Please let me complete the research summary first."
            
        return {
            **state,
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "answer_node",
            "current_step": "answer_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = generate_final_answer(
        params["research_topic"],
        params["summary"],
        params["sources"]
    )

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result.get("success"):
        error_detail = result.get("error", "Could not generate final answer")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User's research topic: '{params["research_topic"]}'
        Respond in one clear sentence. Offer to try again or adjust approach.
        """
        try:
            user_request = llm.invoke([HumanMessage(content=request_prompt)]).content.strip()
        except:
            user_request = f"I had trouble generating the final report. Would you like me to try again?"

        return {
            **state,
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "answer_node",
            "current_step": "answer_node"
        }

    # 4. Success — update state and END
    return {
        **state,
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "final_answer": result.get("answer", "")
        },
        "messages": [
            *state.get("messages", []),
            AIMessage(content=result["answer"])
        ],
        "is_complete": True,  # ✅ Explicitly set completion
        "current_step": "answer_node"
    }


# --- EDGE ROUTING (PURE FUNCTIONS) ---

def route_from_chat(state: Dict[str, Any]) -> str:
    """
    Routes from chat_node.
    - If pending_workflow_step exists and needs_user_input is False, resume workflow
    - Else start fresh with validation_node
    """
    if state.get("pending_workflow_step") and not state.get("needs_user_input", False):
        return state["pending_workflow_step"]  # Resume workflow
    return "validation_node"  # Start fresh


def route_from_validation(state: Dict[str, Any]) -> str:
    """
    Routes from validation_node.
    - If needs_user_input, go to chat_node
    - Else continue to search_node
    """
    return "chat_node" if state.get("needs_user_input") else "search_node"


def route_from_search(state: Dict[str, Any]) -> str:
    """
    Routes from search_node.
    - If needs_user_input, go to chat_node
    - Else continue to summarize_node
    """
    return "chat_node" if state.get("needs_user_input") else "summarize_node"


def route_from_summarize(state: Dict[str, Any]) -> str:
    """
    Routes from summarize_node.
    - If needs_user_input, go to chat_node
    - Else continue to answer_node
    """
    return "chat_node" if state.get("needs_user_input") else "answer_node"


def route_from_answer(state: Dict[str, Any]) -> str:
    """
    Routes from answer_node.
    - If is_complete, END the flow
    - Else go to chat_node
    """
    return "END" if state.get("is_complete") else "chat_node"


# --- WORKFLOW EXECUTOR ---

def create_workflow_graph():
    """
    Creates and compiles the research workflow graph.
    """
    workflow = StateGraph(WorkflowState)

    # Add all nodes
    workflow.add_node("chat_node", chat_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("summarize_node", summarize_node)
    workflow.add_node("answer_node", answer_node)

    # Set entry point
    workflow.set_entry_point("chat_node")

    # Add conditional edges
    workflow.add_conditional_edges("chat_node", route_from_chat)
    workflow.add_conditional_edges("validation_node", route_from_validation)
    workflow.add_conditional_edges("search_node", route_from_search)
    workflow.add_conditional_edges("summarize_node", route_from_summarize)
    workflow.add_conditional_edges("answer_node", route_from_answer)

    return workflow.compile()


# --- RUN ---

if __name__ == "__main__":
    print("🔬 Starting Research Agent Workflow...")

    # --- INITIAL STATE ---
    initial_state = {
        "messages": [],
        "workflow_params": {},
        "current_step": "chat_node",
        "is_complete": False,
        "needs_user_input": True,
        "input_request": generate_contextual_message(
            "The user has not provided a research topic yet. Start the conversation naturally."
        ),
        "pending_workflow_step": None,
        "total_tokens": 0
    }

    # --- BUILD & RUN GRAPH ---
    graph = create_workflow_graph()
    final_state = graph.invoke(initial_state)

    # --- OUTPUT ---
    print("\n\n" + "="*50)
    print("FINAL CHAT HISTORY")
    print("="*50)
    for msg in final_state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"\n[{role}]: {msg.content}")

    print("\n\n" + "="*50)
    print("FINAL RESEARCH REPORT")
    print("="*50)
    if final_state["workflow_params"].get("final_answer"):
        print(final_state["workflow_params"]["final_answer"])
    else:
        print("No report generated.")

    print(f"\n\n📊 Total tokens used: {final_state['total_tokens']}")
