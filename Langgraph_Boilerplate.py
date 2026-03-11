from typing import Dict, TypedDict, List, Any, Optional
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
import json
from langgraph.graph import StateGraph

# --- LLM Setup ---
llm = ChatOpenAI(
    model="r1a1",
    openai_api_key="sk-KHWFQXonSK3yMay5GFP8AA",
    openai_api_base="http://195.201.195.237:4000"
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
def validate_product_name(product_name: str) -> Dict[str, bool]:
    """Simulated lightweight validator"""
    print("\nProduct name:", product_name)
    is_valid = bool(product_name and product_name.strip())
    return {"is_valid": is_valid}

def search_product(product_name: str) -> Dict[str, Any]:
    """
    PURE LLM TOOL.
    Catalog is hardcoded inside.
    LLM decides if product exists.
    No logic, no parsing, no fallback — raw LLM response (structured).
    """

    CATALOG = [
        {"name": "iPhone 15", "price": 999},
        {"name": "iPhone 15 Pro", "price": 1199},
        {"name": "MacBook Air M2", "price": 1099},
        {"name": "Dell XPS 13", "price": 999},
        {"name": "Samsung Galaxy S24", "price": 899},
        {"name": "Google Pixel 8", "price": 799},
    ]

    catalog_text = "\n".join([f"- {item['name']}" for item in CATALOG])

    prompt = f"""
    You are a product catalog matcher.

    USER INPUT: "{product_name}"

    AVAILABLE PRODUCTS:
    {catalog_text}

    TASK:
    - If the product exists in the catalog (exact match only), return success=true and the matched product in "results".
    - If it does not exist, return success=false and a short reason in "error".
    - DO NOT correct typos. DO NOT guess. DO NOT invent.
    - ONLY use products listed above.

    RETURN FORMAT (JSON ONLY):
    {{
        "success": true,
        "results": [{{"name": "..."}}],
        "error": null
    }}
    OR
    {{
        "success": false,
        "results": [],
        "error": "Product not found in catalog."
    }}

    RETURN JSON ONLY. NOTHING ELSE. NO EXPLANATIONS.
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        import json
        content = response.content.strip().strip("`").strip()
        parsed = json.loads(content)

        # Return EXACTLY what LLM returned — no modification
        return parsed

    except Exception as e:
        # If LLM fails, return failure — no fallback
        return {
            "success": False,
            "results": [],
            "error": f"Search tool failed: {str(e)}"
        }

def format_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """Simulated result formatter — uses LLM via generate_contextual_message"""
    if not data.get("success"):
        return {"success": False, "error": "No data to format"}
    
    results = data.get("results", [])
    if not results:
        return {"success": False, "error": "Empty results"}

    # Build list of product names for context
    product_names = "\n".join(f"- {r['name']}" for r in results)
    
    # Reuse your existing LLM message generator
    context = f"The following products were found:\n{product_names}\n\nPresent them to the user in a friendly, natural way."
    formatted_output = generate_contextual_message(context)
    
    return {"success": True, "formatted_output": formatted_output}

def generate_contextual_message(context: str) -> str:
    """
    Uses LLM to generate a natural, user-friendly message based on context.
    """
    prompt = f"""
You are a friendly, helpful assistant in a product search workflow.

CONTEXT:
{context}

TASK:
- Generate a natural, conversational message to show to the user.
- Keep it short, polite, and clear.
- Do NOT use markdown, backticks, or JSON. Just plain text.
- Do NOT explain your reasoning — only return the message.

EXAMPLE:
Context: "User provided an empty product name."
Message: "Hmm, I didn’t catch that. Could you tell me what product you’re looking for?"

RETURN MESSAGE ONLY:
"""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip().strip("`").strip()
    except Exception as e:
        # Fallback if LLM fails
        return "I'm having trouble responding. Could you try again?"
     
# --- NODE EXECUTION LOGIC ---
def chat_node(state: Dict[str, Any]) -> Dict[str, Any]:
    if state["needs_user_input"]:
        print(f"\n[Assistant]: {state['input_request']}")
        user_input = input("[User]: ")
        state["messages"].append(HumanMessage(content=user_input))

        prompt = f"""
You are a strict product name extractor.

TASK:
- Read the user input and extract ONLY the product name as a string.
- If you can extract a clear product name, return: {{"product_name": "extracted name"}}
- If you CANNOT extract any product name, return: {{"product_name": null}}
- Return RAW JSON ONLY — NO MARKDOWN, NO BACKTICKS, NO EXPLANATIONS, NO EXTRA FIELDS.

EXAMPLE 1:
User said: "I want iPhone 15"
RETURN:
{{"product_name": "iPhone 15"}}

EXAMPLE 2:
User said: "blabla foo bar"
RETURN:
{{"product_name": null}}

User said: "{user_input}"

RETURN JSON ONLY:
"""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            print("\nLLMs response:", response)
            state["total_tokens"] += response.response_metadata.get("token_usage", {}).get("total_tokens", 0)

            parsed = json.loads(response.content)

            # Extract product_name
            product_name = parsed.get("product_name")
            print("\nProduct name444:", product_name)
            if isinstance(product_name, str) and product_name.strip():
                # ✅ Success — update workflow_params
                state["workflow_params"]["product_name"] = product_name.strip()
                state["messages"].append(AIMessage(content="Product found. Processing your request..."))
                state["needs_user_input"] = False
                state["input_request"] = ""
                state["pending_workflow_step"] = ""  # Clear it
            else:
                # ❌ product_name is null, not string, or empty
                raise ValueError("Product name is null or invalid")

        except Exception as e:
            print(f"[System]: LLM parsing failed: {e}")
            fallback_msg = generate_contextual_message("The user’s input did not contain a recognizable product name. Ask them to rephrase.")
            state["messages"].append(AIMessage(content=fallback_msg))
            # → Keep asking in chat_node — but set pending step + update message
            state["input_request"] = fallback_msg
            state["pending_workflow_step"] = "chat_node"  # STAY HERE — don't bounce to other nodes
            # → Do NOT update workflow_params
            # → Do NOT set needs_user_input=False → user retries

    return state


# ✅ NEW FORMAT NODES BELOW — REPLACING OLD ONES


def validation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["product_name"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to provide this information.
        Keep it to one sentence. Avoid technical jargon.
        Example: "Could you tell me which product you're looking for?"
        """
        try:
            request = llm.invoke(request_prompt).content.strip()
        except:
            request = f"Please specify the product name."
            
        return {
            **state,  # 👈 Preserve everything
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "validation_node",
            "current_step": "validation_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = validate_product_name(params["product_name"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result["is_valid"]:
        error_detail = "Product name is empty or invalid."
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User is trying to validate a product name: '{params["product_name"]}'
        Respond in one clear sentence. Offer solution or alternative.
        """
        try:
            user_request = llm.invoke(request_prompt).content.strip()
        except:
            user_request = f"Sorry, we couldn't recognize '{params['product_name']}'. Could you try a different product name?"

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
            "validated_product_name": params["product_name"]
        },
        "current_step": "validation_node",
        "messages": [
            *state.get("messages", []),
            AIMessage(content="Validation Successful !")
        ]
    }


def search_node(state: Dict[str, Any]) -> Dict[str, Any]:
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["product_name"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to provide this information.
        Keep it to one sentence. Avoid technical jargon.
        Example: "I need to know what product to search for — could you tell me?"
        """
        try:
            request = llm.invoke(request_prompt).content.strip()
        except:
            request = f"Please specify the product name to search."
            
        return {
            **state,
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "search_node",
            "current_step": "search_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = search_product(params["product_name"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result["success"]:
        error_detail = result.get("error", "Product not found in catalog.")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User searched for product: '{params["product_name"]}'
        Respond in one clear sentence. Offer solution or alternative.
        """
        try:
            user_request = llm.invoke(request_prompt).content.strip()
        except:
            user_request = f"We couldn't find '{params['product_name']}'. Could you try a different name or check spelling?"

        return {
            **state,
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "search_node",
            "current_step": "search_node"
        }

    # 4. Success — update state
    return {
        **state,
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "search_result": result,
            "found_product": result.get("results", [{}])[0] if result.get("results") else {}
        },
        "current_step": "search_node",
        "messages": [
            *state.get("messages", []),
            AIMessage(content="Product found Successful !")
        ]
    }

def result_node(state: Dict[str, Any]) -> Dict[str, Any]:
    params = state["workflow_params"]
    
    # 1. Validate required inputs
    required = ["search_result"]
    missing = [k for k in required if not params.get(k)]
    
    if missing:
        request_prompt = f"""
        Required parameters missing: {', '.join(missing)}.
        Politely ask the user to ensure a product was found first.
        Keep it to one sentence. Avoid technical jargon.
        Example: "I need to find a product first before showing results — could you help me search again?"
        """
        try:
            request = llm.invoke(request_prompt).content.strip()
        except:
            request = f"Please search for a product before viewing results."
            
        return {
            **state,
            "needs_user_input": True,
            "input_request": request,
            "pending_workflow_step": "result_node",
            "current_step": "result_node"
        }

    # 2. EXECUTE PYTHON TOOL
    result = format_result(params["search_result"])

    # 3. Handle tool errors — use LLM to generate user-friendly message
    if not result["success"]:
        error_detail = result.get("error", "Unable to format results.")
        
        request_prompt = f"""
        Convert this technical error into a polite, helpful user request:
        Error: {error_detail}
        Context: User is trying to view formatted results for product search.
        Respond in one clear sentence. Offer solution or alternative.
        """
        try:
            user_request = llm.invoke(request_prompt).content.strip()
        except:
            user_request = f"We had trouble displaying the results. Would you like to try again or adjust your request?"

        return {
            **state,
            "needs_user_input": True,
            "input_request": user_request,
            "pending_workflow_step": "result_node",
            "current_step": "result_node"
        }

    # 4. Success — update state
    return {
        **state,
        "needs_user_input": False,
        "workflow_params": {
            **params,
            "formatted_result": result.get("formatted_output", ""),
            "result_raw": result
        },
        "messages": [
            *state.get("messages", []),
            AIMessage(content=result["formatted_output"])
        ],
        "is_complete": True,  # ✅ Explicitly set completion
        "current_step": "result_node"
    }


# --- EDGE ROUTING (PURE FUNCTIONS) ---
def route_from_chat(state: Dict[str, Any]) -> str:
    if state.get("pending_workflow_step") and not state.get("needs_user_input", False):
        return state["pending_workflow_step"]  # Resume workflow
    return "validation_node"  # Start fresh

def route_from_validation(state: Dict[str, Any]) -> str:
    return "chat_node" if state.get("needs_user_input") else "search_node"

def route_from_search(state: Dict[str, Any]) -> str:
    return "chat_node" if state.get("needs_user_input") else "result_node"

def route_from_result(state: Dict[str, Any]) -> str:
    return "END" if state.get("is_complete") else "chat_node"

# --- NODE REGISTRY ---
NODES = {
    "chat_node": chat_node,
    "validation_node": validation_node,          # ✅ Updated
    "search_node": search_node,                  # ✅ Updated
    "result_node": result_node                   # ✅ Updated
}

EDGES = {
    "chat_node": route_from_chat,
    "validation_node": route_from_validation,
    "search_node": route_from_search,
    "result_node": route_from_result
}

# --- WORKFLOW EXECUTOR ---
def create_workflow_graph():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("chat_node", chat_node)
    workflow.add_node("validation_node", validation_node)
    workflow.add_node("search_node", search_node)
    workflow.add_node("result_node", result_node)

    workflow.set_entry_point("chat_node")

    workflow.add_conditional_edges("chat_node", route_from_chat)
    workflow.add_conditional_edges("validation_node", route_from_validation)
    workflow.add_conditional_edges("search_node", route_from_search)
    workflow.add_conditional_edges("result_node", route_from_result)

    return workflow.compile()

# --- RUN ---
if __name__ == "__main__":
    print("🚀 Starting Product Search Workflow...")

    # --- INITIAL STATE ---
    initial_state = {
        "messages": [],
        "workflow_params": {
            "product_name": "",
            "search_result": {}
        },
        "current_step": "chat_node",
        "is_complete": False,
        "needs_user_input": True,
        "input_request": generate_contextual_message("The user has not provided any product name yet. Start the conversation naturally."),
        "pending_workflow_step": None,
        "total_tokens": 0
    }

    # --- BUILD & RUN GRAPH ---
    graph = create_workflow_graph()
    final_state = graph.invoke(initial_state)

    # --- OUTPUT ---
    print("\n\n--- FINAL CHAT HISTORY ---")
    for msg in final_state["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"[{role}]: {msg.content}")

    print(f"\n📊 Total tokens used: {final_state['total_tokens']}")
