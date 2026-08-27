# TripMate AI — Multi-Agent Travel Planner

TripMate AI is a state-of-the-art multi-agent travel-planning assistant built using **LangGraph**, **Model Context Protocol (MCP)**, **FastAPI**, and a **Human-in-the-Loop (HITL)** design pattern. It integrates multiple specialized agents under a **Supervisor Agent** with built-in input **Guardrails** to ensure a safe, coordinated, and customizable travel-planning experience.

---

## Key Features

*   **Parallel Multi-Agent Architecture**: Orchestrated via LangGraph. Independent specialist agents (**Flights**, **Hotels**, and **Weather**) execute concurrently in parallel (fan-out/fan-in), cutting execution latency by ~50%.
*   **Sequential Feasibility & Integration**: **Budget Agent** consumes merged specialist findings, followed by **Itinerary Agent** crafting the structured draft.
*   **Supervisor Routing**: A central router dynamically selects which specialist agents are needed based on user intent.
*   **Production Guardrails & Real-Time Alerting**: LLM-based intent validation with fail-open safety, sliding window fallback tracking, and error spike alerting via `/api/guardrail/metrics`.
*   **Human-in-the-Loop (HITL) with 2-Min Auto-Approval**: Pauses execution via LangGraph `interrupt()` for human review and feedback. Includes an automated 2-minute countdown timer that automatically finalizes the draft if the user steps away without manual action.
*   **Live MCP Servers**: Integrates Tavily (web search), remote OpenWeather servers, and AviationStack (flight schedules) via the Model Context Protocol.
*   **Modern Web UI**: Clean, interactive glassmorphic UI for real-time collaboration with the planning workflow, live guardrail status badge, and countdown timer.

---

## Architecture & Component Flow

```mermaid
graph TD
    User([User Request]) --> Guardrail{Input Guardrail}
    Guardrail -- Blocked --> BlockedAgent[Guardrail Blocked Node] --> FinalResponse([Final Output])
    Guardrail -- Allowed / Fallback --> Supervisor[Supervisor Agent]
    
    subgraph Parallel Execution [Concurrent Fan-Out]
        Supervisor --> Flight[Flight Agent]
        Supervisor --> Hotel[Hotel Agent]
        Supervisor --> Weather[Weather Agent]
    end
    
    subgraph Sequential Execution [Fan-In Pipeline]
        Flight & Hotel & Weather --> Budget[Budget Agent]
        Budget --> Itinerary[Itinerary Agent]
        Itinerary --> HITL{Human Approval / 2-Min Auto Timeout}
    end
    
    HITL -- Revision Feedback --> FinalAgent[Final Response Agent]
    HITL -- Approved / Auto-Timeout --> FinalAgent --> FinalResponse
```

1. **Input Guardrail**: Checks if the query is travel-related. Fail-open with real-time fallback tracking and alerting. Unrelated queries are safely blocked.
2. **Supervisor Agent**: Parses constraints and triggers required specialist branches.
3. **Parallel Specialist Agents**:
   - `flight_agent`: Retrieves airport, airline, and flight information concurrently.
   - `hotel_agent`: Explores hotels and neighborhoods via Tavily concurrently.
   - `weather_agent`: Checks current weather and forecasts concurrently.
4. **Sequential Agents**:
   - `budget_agent`: Evaluates financial feasibility using merged parallel specialist findings.
   - `itinerary_agent`: Synthesizes all specialist outputs into a comprehensive draft.
5. **Human Approval (HITL)**: The user reviews the draft. If approved (or if 2 minutes elapse without interaction), the draft is polished into the final plan. If rejected, revision feedback is applied.

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

# Live Search, Weather & Flight APIs
TAVILY_API_KEY="your-tavily-api-key"
AVIATIONSTACK_API_KEY="your-aviationstack-api-key"
OPENWEATHER_API_KEY="your-openweather-api-key"
OPENWEATHER_MCP_URL="" # Optional: Custom remote OpenWeather MCP server URL

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

---

## API Endpoints

*   `POST /api/travel`: Submits a prompt or resumes a thread.
    - **Payload**: `{ "message": "Plan a trip to Japan...", "thread_id": null }`
*   `POST /api/travel/approve`: Approves the draft or sends revision feedback.
    - **Payload**: `{ "thread_id": "thread-uuid", "approved": true, "feedback": "" }`
*   `GET /api/guardrail/metrics`: Real-time guardrail telemetry, sliding window fallback rates, recent event audit trail, and active alert status.
*   `POST /api/guardrail/metrics/reset`: Resets guardrail counters and alert history.
*   `GET /health`: Status check API showing loaded features and live guardrail alert state.

---

## Troubleshooting & Notes

*   **AviationStack MCP Dependency Issue**: The default `aviationstack-mcp` package requires version `< 2.0.0` of the python `mcp` library. In `mcp_client.py`, the arguments are configured to force-install the matching dependency via `uvx --with "mcp<2.0.0" aviationstack-mcp`.
*   **Database Mode**: Make sure PostgreSQL is running, as LangGraph uses `PostgresSaver` to persist state across Human-in-the-Loop execution threads.
