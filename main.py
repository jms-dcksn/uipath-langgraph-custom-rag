#from uipath_langchain.chat import UiPathChat
from pydantic import BaseModel
from langchain.messages import HumanMessage, AnyMessage, ToolMessage, AIMessage
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import tools_condition
from langchain.chat_models import init_chat_model
from langchain_classic.tools.retriever import create_retriever_tool
from pydantic import BaseModel, Field
from uipath_langchain.retrievers import ContextGroundingRetriever
from typing import Annotated, TypedDict, Literal
import operator
import os


llm = init_chat_model(model="gpt-4.1-mini")
grader_llm = init_chat_model(model="gpt-4o")

class GraphInput(BaseModel):
    query: str

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    response: str

class GraphOutput(BaseModel):
    response: str

class BinaryGrade(BaseModel):
    binary_grade: str=Field(
        description="'yes' or 'no' based on whether the response is relevant"
        )

retriever = ContextGroundingRetriever(index_name = "ClaimsPC_SOP", folder_path="Shared/FINS")
retriever_tool = create_retriever_tool(
    retriever,
    "ClaimsProcessingStandardOperatingProcedures",
   """
   Use this tool to search for details on handling property and casualty claims and eligibility requirements.
   """
)

# Create a custom retrieve node function
def retrieve_node(state: GraphState):
    """Execute the retriever tool based on tool calls in the last message."""
    messages = state["messages"]
    last_message = messages[-1]
    
    # Check if the last message has tool calls
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_calls = last_message.tool_calls
        tool_messages = []
        
        for tool_call in tool_calls:
            # Handle both dict and ToolCall object formats
            tool_name = tool_call.get("name") if isinstance(tool_call, dict) else getattr(tool_call, "name", None)
            tool_args = tool_call.get("args") if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            tool_id = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
            
            if tool_name == retriever_tool.name:
                # Execute the tool
                result = retriever_tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tool_id,
                    )
                )
        
        return {"messages": tool_messages}
    
    return {"messages": []}

def initialize(input: GraphInput) -> GraphState:
    return {"messages": [HumanMessage(content=input.query)]}

def generate_query_or_respond(state: GraphState):
    """LLM will decide to call context grounding or directly respond"""
    print(retriever_tool.name)
    response = llm.bind_tools([retriever_tool]).invoke(state["messages"])
    return {"messages": [response]}

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

def grade_documents(state: GraphState) -> Literal["generate_answer","rewrite_question"]:
    query = state["messages"][0].content
    # Get the last tool message which contains the retrieved context
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    context = tool_messages[-1].content if tool_messages else ""

    prompt = GRADE_PROMPT.format(question=query, context=context)
    response = (
        grader_llm
        .with_structured_output(BinaryGrade)
        .invoke([{"role": "user", "content": prompt}])
        )
    score = response.binary_grade
    if score == 'yes':
        return "generate_answer"
    else:
        return "rewrite_question"


REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: GraphState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}


GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: GraphState):
    """Generate an answer."""
    question = state["messages"][0].content
    # Get the last tool message which contains the retrieved context
    tool_messages = [msg for msg in state["messages"] if isinstance(msg, ToolMessage)]
    context = tool_messages[-1].content if tool_messages else ""
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}

def output_node(state: GraphState) -> GraphOutput:
    """Final output node that returns GraphOutput."""

    return GraphOutput(response=state["messages"][-1].content)

builder = StateGraph(state_schema=GraphState, input=GraphInput, output=GraphOutput)

builder.add_node("initialize", initialize)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate_query_or_respond", generate_query_or_respond)
builder.add_node("rewrite_question", rewrite_question)
builder.add_node("generate_answer", generate_answer)
builder.add_node("output", output_node)

builder.add_edge(START, "initialize")
builder.add_edge("initialize", "generate_query_or_respond")
builder.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {
        "tools": "retrieve",
        "__end__": "output",
    }
)
builder.add_edge("rewrite_question", "generate_query_or_respond")
# Edges taken after the `action` node is called.
builder.add_conditional_edges(
    "retrieve",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
    }
)
builder.add_edge("generate_answer", "output")
builder.add_edge("output", END)


graph = builder.compile()
