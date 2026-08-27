import os
import certifi
import time
import logging
import datetime
from collections import deque
import threading
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

logger = logging.getLogger("tripmate.guardrail")

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import psycopg
from psycopg.rows import dict_row
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    aviation_mcp_call,
    extract_destination,
    forecast_mcp_search,
    weather_mcp_search,
)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if "sslmode=" not in database_url:
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")

# =========================
# LLM - original model kept
# =========================
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=GROQ_API_KEY,
)

# =========================
# Guardrail Monitoring & Alerting
# =========================
class GuardrailMonitor:
    """
    Production-grade guardrail monitoring and fallback tracker.
    Tracks allowed, blocked, and fallback events in real-time with
    a rolling window, computes fallback rates, and triggers alerts
    when error/fallback rates spike above defined thresholds.
    """
    def __init__(
        self,
        window_size: int = 100,
        alert_threshold: float = 0.20,
        min_sample_size: int = 5,
    ):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        self.min_sample_size = min_sample_size
        self.lock = threading.Lock()
        self.total_requests = 0
        self.allowed_count = 0
        self.blocked_count = 0
        self.fallback_count = 0
        self.recent_events: deque = deque(maxlen=window_size)
        self.alert_history: deque = deque(maxlen=50)
        self.current_alert: dict[str, Any] | None = None

    def record_event(
        self,
        status: str,  # "allowed", "blocked", "fallback"
        latency_ms: float,
        query: str,
        reason: str = "",
        error: str | None = None,
    ) -> dict[str, Any]:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        event = {
            "timestamp": timestamp,
            "status": status,
            "latency_ms": round(latency_ms, 2),
            "query_snippet": query[:120],
            "reason": reason,
            "error": error,
        }

        with self.lock:
            self.total_requests += 1
            if status == "allowed":
                self.allowed_count += 1
            elif status == "blocked":
                self.blocked_count += 1
            elif status == "fallback":
                self.fallback_count += 1

            self.recent_events.append(event)
            self._check_alerts()

        if status == "fallback":
            logger.warning(
                "Guardrail fallback event triggered! Error: %s | Query: %s",
                error,
                query[:80],
            )

        return event

    def _check_alerts(self):
        if len(self.recent_events) < self.min_sample_size:
            self.current_alert = None
            return

        window_fallbacks = sum(
            1 for e in self.recent_events if e["status"] == "fallback"
        )
        window_total = len(self.recent_events)
        rate = window_fallbacks / window_total

        if rate >= self.alert_threshold:
            alert = {
                "triggered_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "level": "CRITICAL" if rate >= 0.40 else "WARNING",
                "fallback_rate": round(rate * 100, 1),
                "threshold": round(self.alert_threshold * 100, 1),
                "sample_size": window_total,
                "fallback_count": window_fallbacks,
                "message": (
                    f"Guardrail fallback rate spike detected: {rate * 100:.1f}% "
                    f"(threshold: {self.alert_threshold * 100:.1f}%) across last "
                    f"{window_total} requests."
                ),
            }
            self.current_alert = alert
            self.alert_history.append(alert)
            logger.error("GUARDRAIL ALERT TRIGGERED: %s", alert["message"])
        else:
            self.current_alert = None

    def get_metrics(self) -> dict[str, Any]:
        with self.lock:
            window_total = len(self.recent_events)
            window_fallbacks = sum(
                1 for e in self.recent_events if e["status"] == "fallback"
            )
            window_fallback_rate = (
                (window_fallbacks / window_total) if window_total > 0 else 0.0
            )

            return {
                "total_requests": self.total_requests,
                "allowed_count": self.allowed_count,
                "blocked_count": self.blocked_count,
                "fallback_count": self.fallback_count,
                "lifetime_fallback_rate": round(
                    (self.fallback_count / self.total_requests * 100)
                    if self.total_requests > 0
                    else 0.0,
                    1,
                ),
                "window_size": self.window_size,
                "window_requests": window_total,
                "window_fallback_count": window_fallbacks,
                "window_fallback_rate": round(window_fallback_rate * 100, 1),
                "alert_threshold_percent": round(self.alert_threshold * 100, 1),
                "is_alerting": self.current_alert is not None,
                "current_alert": self.current_alert,
                "recent_events": list(self.recent_events)[-10:],
                "recent_alerts": list(self.alert_history)[-5:],
            }

    def reset(self):
        with self.lock:
            self.total_requests = 0
            self.allowed_count = 0
            self.blocked_count = 0
            self.fallback_count = 0
            self.recent_events.clear()
            self.alert_history.clear()
            self.current_alert = None


guardrail_monitor = GuardrailMonitor()


def get_guardrail_metrics() -> dict[str, Any]:
    return guardrail_monitor.get_metrics()


# =========================
# State - original fields kept, parallel reducers and guardrail status added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    guardrail_status: str  # "passed", "blocked", "fallback"
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Specialist results (Flight, Hotel, Weather run in parallel)
    flight_results: str
    hotel_results: str
    weather_results: str
    itinerary: str

    # Budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    # Reducer allows concurrent specialists to report LLM calls without collisions
    llm_calls: Annotated[int, operator.add]


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

PARALLEL_SPECIALISTS = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
]

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    return json.loads(text[start : end + 1])


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    guardrail_start = time.perf_counter()

    guardrail_prompt = f"""
Determine whether the following request belongs to travel planning or travel
information. Valid requests can include destinations, flights, hotels, weather,
budgets, visas, transportation, sightseeing, food, packing, or itineraries.

Block clearly unrelated requests and requests asking for harmful or illegal
instructions. Do not block a valid travel request merely because some details
are missing.

Return strict JSON only:
{{
  "allowed": true,
  "reason": ""
}}

User request:
{query}
"""

    llm_calls = 0
    guardrail_status = "passed"

    # Fail open on parser/model errors so a temporary JSON-format issue does not
    # break the original travel-planning behavior, and track fallbacks for alerting.
    try:
        guardrail_raw = _llm_text(
            "You are the input guardrail for a travel-planning application. "
            "Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed = bool(guardrail_result.get("allowed", True))
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
        latency_ms = (time.perf_counter() - guardrail_start) * 1000

        if allowed:
            guardrail_status = "passed"
            guardrail_monitor.record_event(
                status="allowed",
                latency_ms=latency_ms,
                query=query,
                reason=guardrail_reason,
            )
        else:
            guardrail_status = "blocked"
            guardrail_monitor.record_event(
                status="blocked",
                latency_ms=latency_ms,
                query=query,
                reason=guardrail_reason,
            )
    except Exception as exc:
        latency_ms = (time.perf_counter() - guardrail_start) * 1000
        print(f"Guardrail fallback used: {exc}")
        allowed = True
        guardrail_status = "fallback"
        guardrail_reason = "Guardrail validation fallback allowed the request."
        guardrail_monitor.record_event(
            status="fallback",
            latency_ms=latency_ms,
            query=query,
            reason=guardrail_reason,
            error=str(exc),
        )

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "guardrail_status": "blocked",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""
You are the supervisor of a multi-agent travel-planning system.
Choose only the specialist agents needed for the request.

If the origin is not specified in the User request, default the 
"origin" in trip_constraints to "India".

Available agents:
- flight_agent: flights, airports, airlines, routes, airfare, or booking advice
- hotel_agent: hotels, accommodation, neighborhoods, or places to stay
- weather_agent: weather, climate, season, forecast, or packing advice
- budget_agent: cost, affordability, price limits, or budget feasibility
- itinerary_agent: creates the integrated travel plan and must always be included

Return strict JSON only using this schema:
{{
  "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
  "trip_constraints": {{
    "destination": "",
    "origin": "",
    "duration": "",
    "budget": "",
    "travel_style": "",
    "special_preferences": []
  }},
  "reasoning": ""
}}

User request:
{query}
"""

    try:
        supervisor_raw = _llm_text(
            "You route work to travel specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "guardrail_status": guardrail_status,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent - original behavior kept
# =========================
FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": 1,
    }


# =========================
# Hotel Agent - original behavior kept (runs in parallel)
# =========================
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"

    try:
        hotel_results = asyncio.run(tavily_mcp_search(query))
    except Exception as exc:
        print(
            f"HOTEL AGENT MCP ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )
        hotel_results = (
            "Live hotel search is temporarily unavailable. "
            "Provide general accommodation and neighborhood "
            "guidance based on the destination and clearly "
            "label it as non-live advice."
        )

    return {
        "hotel_results": hotel_results,
        "messages": [AIMessage(content="Hotel information processed.")],
        "llm_calls": 1,
    }


# =========================
# Weather Agent - original behavior kept (runs in parallel)
# =========================
def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])

    try:
        weather_data = asyncio.run(weather_mcp_search(city))
        forecast_data = asyncio.run(forecast_mcp_search(city))

        weather_results = f"""
Current Weather:
{weather_data}

Forecast:
{forecast_data}
"""
    except Exception as exc:
        print(
            f"WEATHER AGENT MCP ERROR: {type(exc).__name__}: {exc}",
            flush=True,
        )
        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [AIMessage(content="Weather information processed.")],
        "llm_calls": 0,
    }


# =========================
# Budget Agent - sequential specialist after parallel fan-in
# =========================
def budget_agent(state: TravelState):
    # If supervisor did not select budget_agent, skip calculation
    if "budget_agent" not in state.get("selected_agents", []):
        return {"budget_results": ""}

    prompt = f"""
Analyze whether this trip is realistic for the user's budget.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Return:
1. Estimated cost categories
2. Budget risk areas
3. Money-saving suggestions
4. Overall feasibility

If exact live prices are unavailable, clearly label estimates as approximate.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": 1,
    }


# =========================
# Itinerary Agent - sequential specialist after budget analysis
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""
Create a complete travel itinerary.

User Query:
{state['user_query']}

Trip Constraints:
{state.get('trip_constraints', {})}

Flight Results:
{state.get('flight_results', '')}

Hotel Results:
{state.get('hotel_results', '')}

Weather Results:
{state.get('weather_results', '')}

Budget Results:
{state.get('budget_results', '')}

Make the itinerary practical, budget-aware, and easy to follow.
Create a clear draft that is ready for human review.
"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - original format kept, HITL feedback added
# =========================
def final_agent(state: TravelState):
    if state.get("approved", False):
        review_instruction = (
            "The user approved the draft. Preserve its decisions while polishing it."
        )
    else:
        review_instruction = f"""
The user requested a revision. Apply this feedback carefully:
{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}
"""

    final_prompt = f"""
Generate the final travel response for the user.

Human Review:
{review_instruction}

User Request:
{state['user_query']}

Supervisor Constraints:
{state.get('trip_constraints', {})}

Flights:
{state.get('flight_results', '')}

Hotels:
{state.get('hotel_results', '')}

Weather:
{state.get('weather_results', '')}

Budget Analysis:
{state.get('budget_results', '')}

Draft Itinerary:
{state.get('itinerary', '')}

Format the final answer beautifully using these sections:
1. Trip Summary
2. Flight Information
3. Hotel Suggestions
4. Weather Information
5. Day-by-Day Itinerary
6. Estimated Budget
7. Final Recommendations

Important:
- Be clear and practical.
- Mention that live flight APIs may not provide ticket prices when pricing is unavailable.
- Include weather-based travel advice.
- Keep the response useful for real travel planning.
- Incorporate the human feedback when revision was requested.
"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional AI travel booking assistant."
            ),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": 1,
    }


# =========================
# Graph Routing (Parallel Fan-Out -> Sequential Pipeline)
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
}


def route_from_supervisor(state: TravelState) -> list[str]:
    """
    Parallel fan-out router:
    - If guardrail blocked: routes to guardrail_blocked node.
    - If guardrail allowed: launches selected independent specialists
      (flight, hotel, weather) simultaneously in parallel.
    - If no parallel specialists were selected, routes directly to budget_agent.
    """
    if not state.get("guardrail_allowed", True):
        return ["guardrail_blocked"]

    selected = state.get("selected_agents", [])
    parallel_to_run = [
        agent for agent in PARALLEL_SPECIALISTS if agent in selected
    ]

    if parallel_to_run:
        return parallel_to_run

    return ["budget_agent"]


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor")

# 1. Parallel Fan-Out: supervisor conditionally triggers selected specialist branches
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

# 2. Parallel Fan-In: flight, hotel, and weather converge to budget_agent
graph.add_edge("flight_agent", "budget_agent")
graph.add_edge("hotel_agent", "budget_agent")
graph.add_edge("weather_agent", "budget_agent")

# 3. Sequential Pipeline: budget -> itinerary -> human review -> final response
graph.add_edge("budget_agent", "itinerary_agent")
graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)

# =========================
# PostgreSQL Checkpointer - original persistence kept
# =========================
DATABASE_URL = get_database_url()
_conn = psycopg.connect(
    DATABASE_URL,
    autocommit=True,
    row_factory=dict_row,
)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

travel_graph = graph.compile(checkpointer=checkpointer)


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    interrupt_payload = _interrupt_payload(result)

    if interrupt_payload:
        answer = interrupt_payload.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": interrupt_payload is not None,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            interrupt_payload.get("draft_itinerary", "")
            if interrupt_payload
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "guardrail_status": result.get("guardrail_status", "passed"),
        "guardrail_metrics": guardrail_monitor.get_metrics(),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "guardrail_status": "passed",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
