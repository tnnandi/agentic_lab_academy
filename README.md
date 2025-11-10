# Agentic Lab (Academy Edition)

This project re-implements the original [Agentic Lab](../agentic_lab/README.md) multi-agent research framework using the [Academy](https://github.com/proxystore/academy) middleware. The goal is to preserve the collaborative research–coding workflow while expressing each role as an Academy `Agent` with well-defined asynchronous actions.

## Highlights

- **Academy runtime** – agents communicate through an exchange managed by `academy.manager.Manager` with a `LocalExchangeFactory` backend.
- **Stateful actors** – principal investigator, browsing, research, coding, execution, reviewing, and critic roles are implemented as distinct Academy agents.
- **LLM-driven workflow** – reuses the prompts and utilities from the legacy project with pluggable LLM backends (local Ollama or the ALCF inference endpoints).
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
├── llm.py                    # Sync + async helpers for querying configured LLM backends
├── main.py                   # CLI entry point (mirrors legacy arguments)
├── alcf_inference/           # Globus helper for authenticating with ALCF inference
├── models.py                 # Dataclasses for agent message payloads
├── prompts.py                # Prompt builders reused from the legacy project
├── utils.py                  # PDF/link processing, DuckDuckGo search, persistence
└── workflows/
    ├── __init__.py
    └── orchestrator.py       # Runs the multi-agent workflow via Manager
```

## Running the Framework

1. Install dependencies (Ollama or ALCF credentials required depending on backend):

   ```bash
   pip install -r requirements.txt
   ```

2. Launch the workflow:

   ```bash
   python -m agentic_lab_academy.main --topic "Your topic here" --mode both
   ```

   Command-line options mirror the legacy project: provide PDFs, links, file directories, specify `--quick_search`, and set a `--conda_env` for code execution if needed.

3. Choose the LLM backend in `config.py` by setting `LLM_CONFIG["source"]` to:
   - `ollama` (default) – requires a local Ollama service.
   - `alcf_sophia` or `alcf_metis` – requires prior authentication via `python -m agentic_lab_academy.alcf_inference.inference_auth_token authenticate`.
   - `openrouter` – provide an `OPENROUTER_API_KEY` (plus optional `OPENROUTER_SITE_URL`/`OPENROUTER_APP_NAME`) to use OpenRouter-hosted models such as `openrouter/polaris-alpha`.

4. Follow on-screen prompts to approve the PI-generated plan (and coding plan when requested). Iteration logs, research reports, generated code, and execution outputs are stored under `output_agent/`.

5. For ALCF Sophia or other HPC submissions, run the CLI with `--use_hpc`. The coding agent still writes the Python training script, but the new `HPCAgent` creates a PBS batch script (stored under `workspace_runs/.../hpc_jobs/`) with directives such as `#PBS -A GeomicVar`, `#PBS -l select=1:system=sophia`, `#PBS -l filesystems=home:grand`, `#PBS -l walltime=01:00:00`, and `#PBS -q by-gpu`, submits it via `qsub`, and records the job ID. After submission, HPCAgent polls `qstat` every 10 seconds (configurable via `status_poll_interval`/`status_max_checks` in `hpc_options`) so you can watch the queue state from the CLI. Once the job leaves the queue, HPCAgent reads the generated `hpc_job_iterXX_YY.out/err` files, reasons about any failures, and feeds that feedback back into the coding agent before submitting the next attempt.

## Notes

- All files are created within this directory tree to respect the sandboxed environment.
- The Academy implementation continues to rely on the same LLM prompt engineering as the original project; ensure the selected backend in `config.py` (Ollama or ALCF) is reachable and authenticated.
- The orchestrator uses a `ThreadPoolExecutor` to offload blocking LLM and subprocess calls while keeping inter-agent communication asynchronous.


