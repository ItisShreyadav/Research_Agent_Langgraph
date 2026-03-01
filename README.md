# Research_Agent_Langgraph

A compiled LangGraph where:  The graph always executes in the order: Search → Summarize → Answer .
Intermediate outputs are stored in state.
The agent must not decide its own steps.
The execution path is predefined in the graph.
Each step is a separate LangGraph node with explicit state passing.
