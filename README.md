# Agentic Lab (Academy Edition)

This project re-implements the original [Agentic Lab](../agentic_lab/README.md) multi-agent research framework using the [Academy](https://github.com/proxystore/academy) middleware. The goal is to preserve the collaborative research–coding workflow while expressing each role as an Academy `Agent` with well-defined asynchronous actions.

## Highlights

- **Academy runtime** – agents communicate through an exchange managed by `academy.manager.Manager` with a `LocalExchangeFactory` backend.
- **Stateful actors** – principal investigator, browsing, research, coding, execution, reviewing, and critic roles are implemented as distinct Academy agents.
- **LLM-driven workflow** – reuses the prompts, utilities, and Ollama-backed LLM helpers from the legacy project for topic planning, drafting, coding, and critique.
- **Iterative refinement** – preserves the feedback loop between agents with explicit iteration state, code execution, and critic summaries routed through the manager.
- **Workspace outputs** – reports, code, and execution logs are saved under `./output_agent/` within this project directory for each iteration.

## Repository Layout

```
agentic_lab_academy/
├── README.md
├── requirements.txt
├── pyproject.toml
├── __init__.py
├── academy_agents.py         # Academy Agent subclasses for each role
├── config.py                 # Shared configuration (LLM defaults, etc.)
├── llm.py                    # Sync + async helpers for querying Ollama
├── main.py                   # CLI entry point (mirrors legacy arguments)
├── models.py                 # Dataclasses for agent message payloads
├── prompts.py                # Prompt builders reused from the legacy project
├── utils.py                  # PDF/link processing, DuckDuckGo search, persistence
└── workflows/
    ├── __init__.py
    └── orchestrator.py       # Runs the multi-agent workflow via Manager
```

## Running the Framework

1. Install dependencies (an existing Ollama deployment is required):

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the workflow:

   ```bash
   python -m agentic_lab_academy.main --topic "Your topic here" --mode both
   ```

   Command-line options mirror the legacy project: provide PDFs, links, file directories, specify `--quick_search`, and set a `--conda_env` for code execution if needed.

3. Follow on-screen prompts to approve the PI-generated plan (and coding plan when requested). Iteration logs, research reports, generated code, and execution outputs are stored under `output_agent/`.

## Notes

- All files are created within this directory tree to respect the sandboxed environment.
- The Academy implementation continues to rely on the same LLM prompt engineering as the original project; ensure the Ollama endpoint configured in `config.py` is reachable.
- The orchestrator uses a `ThreadPoolExecutor` to offload blocking LLM and subprocess calls while keeping inter-agent communication asynchronous.

## License

This codebase inherits the licensing terms of the upstream Agentic Lab project.
