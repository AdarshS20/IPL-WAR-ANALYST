"""
Local LLM tool-use wrapper via Ollama: takes a natural-language question
about IPL WAR, lets a locally-running model pick and call the right
tool(s) from tools.py, and returns a conversational answer grounded in
the actual data.

Runs entirely against a local Ollama server -- no API key, nothing sent
to the cloud, nothing to hand out when presenting to someone else.

One-time setup before presenting:
    1. Install Ollama:            https://ollama.com/download
    2. Pull a tool-calling model:  ollama pull llama3.1
    3. Make sure the Ollama app/service is running (it serves
       http://localhost:11434 automatically on most installs).

Swap models with the OLLAMA_MODEL env var, e.g. a lighter one for an
older laptop: `set OLLAMA_MODEL=llama3.2` (after `ollama pull llama3.2`).
"""

import json
import os

import ollama

import tools

MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST")  # None -> client default (localhost:11434)
MAX_TOOL_ROUNDS = 4

SYSTEM_PROMPT = """\
You are an IPL cricket analytics assistant. You answer questions using a \
custom WAR (Wins Above Replacement) model built from CricSheet ball-by-ball \
data, computed separately for batting, bowling, and fielding, then summed \
for a total WAR. Key modelling details, in case they're relevant to the \
question: RUNS_PER_WIN = 10 runs, a player must reach a minimum ball \
threshold (60 balls faced/bowled) to qualify for batting/bowling WAR, and \
overs are split into powerplay/middle/death phases with different \
leverage weights. Fielding WAR uses its own scale (RUNS_PER_WIN = 16, \
minimum 5 matches). Data covers IPL seasons 2008 through the latest \
available season.

You have tools to look up player WAR, batting splits vs pace/spin, bowler \
matchup stats vs left-/right-handed batters, head-to-head player \
comparisons, season leaderboards, and wicketkeeper stats. Always call a \
tool to get numbers rather than guessing them yourself. If a tool reports \
an error (e.g. player not found), explain that plainly to the user and, \
if the error suggests alternatives, offer them. Don't invent stats that \
didn't come from a tool.

Report only the fields a tool actually returned, using their real names \
(economy is not an average; strike_rate is not WAR). Never relabel one \
stat as another and never compute or estimate a stat the tool didn't give \
you. Pass player names exactly as the user wrote them; the tools resolve \
them.

When answering, lead with the direct answer, then back it up with the \
specific numbers (WAR values, strike rate, economy, etc.) that justify it. \
Keep it conversational and concise -- a couple of short paragraphs at most \
unless the user asks for a full breakdown.\
"""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_player_war",
            "description": (
                "Get a player's combined batting, bowling, and fielding WAR. "
                "Omit season for career totals (summed across seasons) with a "
                "per-season breakdown; pass season for a single IPL season."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string", "description": "Player name, e.g. 'V Kohli' or just 'Kohli'."},
                    "season": {"type": "integer", "description": "IPL season year, e.g. 2016. Omit for career totals."},
                },
                "required": ["player"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_batting_splits",
            "description": "Get a batter's runs/balls/strike rate/average against pace or spin bowling.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string", "description": "Batter name."},
                    "bowler_type": {"type": "string", "enum": ["pace", "spin"], "description": "Bowling type to filter deliveries by."},
                    "season": {"type": "integer", "description": "IPL season year. Omit for career."},
                },
                "required": ["player", "bowler_type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_matchup_stats",
            "description": "Get a bowler's economy/wickets against left-handed or right-handed batters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bowler": {"type": "string", "description": "Bowler name."},
                    "batter_hand": {"type": "string", "enum": ["LHB", "RHB"], "description": "LHB = left-handed batter, RHB = right-handed batter."},
                    "season": {"type": "integer", "description": "IPL season year. Omit for career."},
                },
                "required": ["bowler", "batter_hand"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_player_comparison",
            "description": "Compare total WAR (and the batting/bowling/fielding breakdown) of two players, head to head.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player1": {"type": "string"},
                    "player2": {"type": "string"},
                    "season": {"type": "integer", "description": "IPL season year. Omit for career comparison."},
                },
                "required": ["player1", "player2"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_season_leaderboard",
            "description": "Get the top N players in a season ranked by a WAR metric.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["batting_war", "bowling_war", "fielding_war", "total_war"]},
                    "season": {"type": "integer", "description": "IPL season year, required."},
                    "top_n": {"type": "integer", "description": "How many players to return (default 10, max 50)."},
                },
                "required": ["metric", "season"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_keeper_stats",
            "description": "Get wicketkeeper-relevant fielding numbers for a player: byes conceded, stumpings, catches, and whether they were the primary keeper.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player": {"type": "string"},
                    "season": {"type": "integer", "description": "IPL season year. Omit for career totals."},
                },
                "required": ["player"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_player_war": tools.get_player_war,
    "get_batting_splits": tools.get_batting_splits,
    "get_matchup_stats": tools.get_matchup_stats,
    "get_player_comparison": tools.get_player_comparison,
    "get_season_leaderboard": tools.get_season_leaderboard,
    "get_keeper_stats": tools.get_keeper_stats,
}


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama.Client()


def _run_tool(name: str, tool_input: dict) -> dict:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"error": f"Unknown tool '{name}'"}
    try:
        return fn(**tool_input)
    except Exception as e:
        return {"error": str(e)}


def is_available() -> tuple:
    """
    Check whether the local Ollama server is reachable and MODEL is pulled.
    Returns (ok, message) -- message is empty on success, else an
    actionable explanation to show the user.
    """
    try:
        listed = _client().list()
        pulled = {m.model for m in listed.models}
        bare_pulled = {n.split(":")[0] for n in pulled}
        if MODEL in pulled or MODEL.split(":")[0] in bare_pulled:
            return True, ""
        return False, f"Ollama is running, but model '{MODEL}' isn't pulled yet. Run: ollama pull {MODEL}"
    except Exception as e:
        host = OLLAMA_HOST or "http://localhost:11434"
        return False, f"Can't reach Ollama at {host} ({e}). Is the Ollama app/service running?"


def ask(question: str, history: list = None) -> dict:
    """
    Answer one question, optionally continuing a prior conversation.

    `history` is a list of message dicts from a previous call's returned
    "messages" (pass None / [] to start fresh).

    Returns {"answer": str, "messages": list, "tool_calls": list} where
    "messages" should be threaded into the next call to keep context, and
    "tool_calls" is a trace of every tool invoked this turn (name, input,
    output) useful for showing "what data backs this answer" in a UI.
    """
    client = _client()
    messages = list(history) if history else [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": question})

    tool_trace = []
    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat(
            model=MODEL,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            options={"temperature": 0},
        )
        msg = response.message

        assistant_msg = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {"function": {"name": tc.function.name, "arguments": dict(tc.function.arguments)}}
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        if not msg.tool_calls:
            return {"answer": msg.content or "", "messages": messages, "tool_calls": tool_trace}

        for tc in msg.tool_calls:
            name = tc.function.name
            call_input = dict(tc.function.arguments)
            result = _run_tool(name, call_input)
            tool_trace.append({"tool": name, "input": call_input, "output": result})
            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(result, default=str),
            })

    return {
        "answer": "I wasn't able to finish looking that up within the allotted tool calls -- try rephrasing or narrowing the question.",
        "messages": messages,
        "tool_calls": tool_trace,
    }


if __name__ == "__main__":
    import sys

    ok, msg = is_available()
    if not ok:
        print(f"Ollama not ready: {msg}")
        sys.exit(1)

    q = " ".join(sys.argv[1:]) or "Who had the highest total WAR in the 2016 season?"
    result = ask(q)
    print(f"\nQ: {q}\n")
    print(result["answer"])
    if result["tool_calls"]:
        print("\n--- tool calls ---")
        for call in result["tool_calls"]:
            print(f"{call['tool']}({call['input']})")
