# TripMate AI — Multi-Agent Travel Planner

TripMate AI is a state-of-the-art multi-agent travel-planning assistant built using **LangGraph**, **Model Context Protocol (MCP)**, **FastAPI**, and a **Human-in-the-Loop (HITL)** design pattern. It integrates multiple specialized agents under a **Supervisor Agent** with built-in input **Guardrails** to ensure a safe, coordinated, and customizable travel-planning experience.

---

## Key Features

*   **Multi-Agent Architecture**: Orchestrated via LangGraph. Each specialist agent (Flights, Hotels, Weather, Budget, Itinerary) focuses on a single domain.
*   **Supervisor Routing**: A central router dynamically selects which specialist agents are needed based on the user's request.
*   **Input Guardrails**: Validates and filters out non-travel or potentially harmful user requests before they enter the planning graph.
*   **Human-in-the-Loop (HITL)**: Pauses execution to allow users to review, edit, approve, or provide feedback on the draft itinerary before generating the final polished plan.
*   **Live MCP Servers**: Integrates Tavily (web search), custom Weather servers, and AviationStack (flight schedules) via the Model Context Protocol.
*   **Modern Web UI**: Clean, interactive glassmorphic UI for real-time collaboration with the planning workflow.

---

## Architecture & Component Flow

```mermaid
graph TD
    User([User Request]) --> Guardrail{Input Guardrail}
    Guardrail -- Blocked --> BlockedAgent[Guardrail Agent] --> FinalResponse([Final Output])
    Guardrail -- Allowed --> Supervisor[Supervisor Agent]
    
    Supervisor --> RouteMap{Selected Specialists}
    RouteMap --> Flight[Flight Agent]
    RouteMap --> Hotel[Hotel Agent]
    RouteMap --> Weather[Weather Agent]
    RouteMap --> Budget[Budget Agent]
    
    Flight & Hotel & Weather & Budget --> Itinerary[Itinerary Agent]
    Itinerary --> HITL{Human Approval}
    
    HITL -- Revision Feedback --> Supervisor
    HITL -- Approved --> FinalAgent[Final Response Agent] --> FinalResponse
```

1. **Input Guardrail**: Checks if the query is travel-related. Unrelated queries are blocked.
2. **Supervisor Agent**: Parses constraints (origin, destination, budget, etc.) and routes the task to required agents.
3. **Specialist Agents**:
   - `flight_agent`: Retrieves airport, airline, and flight information.
   - `hotel_agent`: Explores hotels and neighborhoods via Tavily.
   - `weather_agent`: Checks current weather and forecasts.
   - `budget_agent`: Evaluates financial feasibility of the trip.
4. **Itinerary Agent**: Consolidates findings into a complete draft.
5. **Human Approval**: The user reviews the draft. If approved, the final plan is generated. If rejected, feedback is fed back to the supervisor.

---

## Prerequisites

*   Python 3.10 or 3.11 (3.11 is recommended)
*   PostgreSQL database (required for LangGraph's persistent state checkpointing)
*   `uv` or `uvx` package manager (optional, but highly recommended for launching MCP servers)

---

## Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/Multi-Agent-System-using-LangGraph-MCP-Supervisor-Guardrails-HITL.git
cd Multi-Agent-System-using-LangGraph-MCP-Supervisor-Guardrails-HITL
```

### 2. Set Up Virtual Environment
Using `venv` (standard):
```bash
python -m venv .venv
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate
```

Using `uv` (faster):
```bash
uv venv
# On Windows
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables (`.env`)
Create a `.env` file in the root directory and populate it with your API keys:
```env
# LLM Provider Key (Groq Llama-3.3-70b-versatile)
GROQ_API_KEY="your-groq-api-key"

# Live Search & Flight APIs
TAVILY_API_KEY="your-tavily-api-key"
AVIATIONSTACK_API_KEY="your-aviationstack-api-key"
OPENWEATHER_API_KEY="your-openweather-api-key"

# LangGraph Checkpointer Database (PostgreSQL)
DATABASE_URL="postgresql://username:password@localhost:5432/dbname"

# Optional: LangSmith Tracing
LANGSMITH_TRACING="true"
LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
LANGSMITH_API_KEY="your-langsmith-api-key"
LANGSMITH_PROJECT="Travel-Agent"
```

---

## Running the Application

### 1. Run FastAPI Web Server
```bash
# Using uvicorn
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.

### 2. Running the Custom Weather MCP Server (Optional/Background)
The Weather agent runs the local MCP weather server dynamically. If you want to run it manually or verify it works:
```bash
python custom_weather_mcp_server.py
```

---

## API Endpoints

*   `POST /api/travel`: Submits a prompt or resumes a thread.
    - **Payload**: `{ "message": "Plan a trip to Japan...", "thread_id": null }`
*   `POST /api/travel/approve`: Approves the draft or sends revision feedback.
    - **Payload**: `{ "thread_id": "thread-uuid", "approved": true, "feedback": "" }`
*   `GET /health`: Status check API showing loaded features.

---

## Troubleshooting & Notes

*   **AviationStack MCP Dependency Issue**: The default `aviationstack-mcp` package requires version `< 2.0.0` of the python `mcp` library. In `mcp_client.py`, the arguments are configured to force-install the matching dependency via `uvx --with "mcp<2.0.0" aviationstack-mcp`.
*   **Database Mode**: Make sure PostgreSQL is running, as LangGraph uses `PostgresSaver` to persist state across Human-in-the-Loop execution threads.
