"""Academy agent implementations for the Agentic Lab workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from academy.agent import Agent, action

try:
    from . import prompts, utils
    from .config import LLM_CONFIG
    from .llm import query_llm_async, query_llm
    from .models import CodeArtifact, CritiqueBundle, ExecutionResult, PlanResult, ResearchArtifact
except ImportError:
    import prompts, utils
    from config import LLM_CONFIG
    from llm import query_llm_async, query_llm
    from models import CodeArtifact, CritiqueBundle, ExecutionResult, PlanResult, ResearchArtifact

__all__ = [
    "PrincipalInvestigatorAgent",
    "BrowsingAgent",
    "ResearchAgent",
    "CodeWriterAgent",
    "CodeExecutorAgent",
    "HPCAgent",
    "CodeReviewerAgent",
    "CriticAgent",
]


@dataclass
class _RunContext:
    working_dir: Path
    iteration: int


class PrincipalInvestigatorAgent(Agent):
    def __init__(self, *, verbose: bool = True, max_rounds: int = 3) -> None:
        super().__init__()
        self.verbose = verbose
        self.max_rounds = max_rounds

    @action
    async def configure(self, verbose: bool | None = None, max_rounds: int | None = None) -> None:
        if verbose is not None:
            self.verbose = verbose
        if max_rounds is not None:
            self.max_rounds = max_rounds

    @action
    async def create_plan(self, sources: str, topic: str, mode: str, changes: str | None = None) -> dict:
        prompt = prompts.get_pi_plan_prompt(sources, topic, mode, changes)
        plan = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["research"])

        reasoning = None
        if changes:
            reasoning_prompt = prompts.get_plan_changes_reasoning_prompt(changes, topic, mode)
            reasoning = await query_llm_async(
                reasoning_prompt, temperature=LLM_CONFIG["temperature"]["research"]
            )

        if self.verbose:
            print("PI Agent generated plan:\n", plan)
            if reasoning:
                print("PI Agent reasoning about changes:\n", reasoning)

        return PlanResult(plan=plan, reasoning=reasoning).to_dict()


class BrowsingAgent(Agent):     #TN: Can add BioMCP here for searching pubmed 
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @action
    async def gather_sources(
        self,
        topic: str,
        pdf_content: str = "",
        links: Sequence[str] | None = None,
        files_dir_content: str = "",
        include_directory_listing: bool = True,
    ) -> str:
        if self.verbose:
            print(f"BrowsingAgent: gathering sources for '{topic}'")

        combined_sources: list[str] = []

        if links:
            link_content = await asyncio.get_running_loop().run_in_executor(
                None, utils.process_links, list(links)
            )
            if link_content:
                combined_sources.append(f"Link Content:\n{link_content}")

        if pdf_content:
            combined_sources.append(f"PDF Content:\n{pdf_content}")

        if files_dir_content:
            combined_sources.append(f"Files Directory Content:\n{files_dir_content}")

        if include_directory_listing:
            current_dir = Path.cwd()
            file_listing = "\n".join(sorted(p.name for p in current_dir.iterdir() if p.is_file()))
            combined_sources.append(
                f"Current Directory Information:\nCurrent Working Directory: {current_dir}\nFiles in current directory:\n{file_listing}"
            )

        formatted_sources = "\n\n".join(combined_sources)

        if self.verbose:
            print("BrowsingAgent assembled sources preview:\n", formatted_sources[:1000])

        return formatted_sources

    @action
    async def quick_search(self, topic: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, utils.quick_duckduckgo_search, topic)


class ResearchAgent(Agent):
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @action
    async def draft_document(
        self, sources: str, topic: str, plan_section: str = "", iteration: int = 0
    ) -> dict:
        prompt = prompts.get_only_research_draft_prompt(sources, topic, plan_section)
        raw_report = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["research"])
        report = utils.clean_report(raw_report)
        if self.verbose:
            print("ResearchAgent draft complete (truncated):\n", report[:800])
        return ResearchArtifact(content=report, iteration=iteration).to_dict()

    @action
    async def improve_document(self, draft: str, feedback: str, iteration: int) -> dict:
        prompt = prompts.get_research_improve_prompt(draft, feedback)
        raw_report = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["research"])
        report = utils.clean_report(raw_report)
        if self.verbose:
            print("ResearchAgent improved draft (truncated):\n", report[:800])
        return ResearchArtifact(content=report, iteration=iteration).to_dict()


class CodeWriterAgent(Agent):
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @action
    async def create_coding_plan(self, sources: str, topic: str, plan_section: str = "") -> str:
        if self.verbose:
            print("CodeWriterAgent: creating coding plan")
        prompt = prompts.get_coding_plan_prompt(sources, topic, plan_section)
        return await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["coding"])

    @action
    async def improve_coding_plan(self, feedback: str, coding_plan: str) -> str:
        if self.verbose:
            print(f"CodeWriterAgent: improving coding plan based on feedback: {feedback}")
        prompt = prompts.get_improved_coding_plan_prompt(feedback, coding_plan)
        return await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["coding"])

    @action
    async def create_code(
        self,
        sources: str,
        topic: str,
        plan_section: str,
        coding_plan: str,
        iteration: int,
    ) -> dict:
        prompt = prompts.get_code_writing_prompt(sources, topic, plan_section, coding_plan)
        response = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["coding"])
        code = utils.extract_code_only(response)
        if self.verbose:
            print("CodeWriterAgent produced code (truncated):\n", code[:80])
            # print("CodeWriterAgent produced code :\n", code)

        return CodeArtifact(code=code, iteration=iteration).to_dict()

    @action
    async def improve_code(self, code: str, feedback: str, iteration: int) -> dict:
        prompt = prompts.get_code_improve_prompt(code, feedback)
        response = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["coding"])
        improved = utils.extract_code_only(response)
        if self.verbose:
            print("CodeWriterAgent improved code (truncated):\n", improved[:80])
            # print("CodeWriterAgent improved code :\n", improved)
        return CodeArtifact(code=improved, iteration=iteration).to_dict()


class CodeExecutorAgent(Agent):
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.execution_counter = 0

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @staticmethod
    def _python_executable(conda_env_path: str | None) -> str:
        if not conda_env_path:
            return "python"
        candidates = [
            Path(conda_env_path) / "bin" / "python",
            Path(conda_env_path) / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return "python"

    def _write_script(self, code: str, working_dir: Path, iteration: int) -> Path:
        scripts_dir = working_dir / "generated_code"
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / f"iteration_{iteration:02d}_{self.execution_counter:02d}.py"
        script_path.write_text(code)
        return script_path

    async def _run_subprocess(self, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
            ),
        )

    @staticmethod
    def _missing_modules(stderr: str) -> list[str]:
        pattern = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
        return list({match.group(1) for match in pattern.finditer(stderr)})

    async def _resolve_packages(self, modules: Iterable[str]) -> list[str]:
        packages: list[str] = []
        for mod in modules:
            prompt = prompts.get_package_resolution_prompt(mod)
            package = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["execution"])
            packages.append(package.strip())
        return packages

    async def _install_packages(self, packages: Sequence[str], python_exe: str) -> tuple[list[str], str]:
        if not packages:
            return [], ""
        pip_cmd = [python_exe, "-m", "pip", "install", *packages]
        result = await self._run_subprocess(pip_cmd, Path.cwd())
        return list(packages), result.stdout + "\n" + result.stderr

    @action
    async def execute_code(
        self,
        code: str,
        working_directory: str,
        iteration: int,
        conda_env_path: str | None = None,
    ) -> dict:
        self.execution_counter += 1
        workdir = Path(working_directory)
        workdir.mkdir(parents=True, exist_ok=True)
        script_path = self._write_script(code, workdir, iteration)
        python_exe = self._python_executable(conda_env_path)

        if self.verbose:
            print(f"CodeExecutorAgent running script {script_path}")

        result = await self._run_subprocess([python_exe, str(script_path)], workdir)
        packages_installed: list[str] = []

        if result.returncode != 0:
            missing = self._missing_modules(result.stderr)
            if missing:
                if self.verbose:
                    print("CodeExecutorAgent detected missing modules:", missing)
                resolved = await self._resolve_packages(missing)
                installed, install_logs = await self._install_packages(resolved, python_exe)
                packages_installed.extend(installed)
                if self.verbose and installed:
                    print("Installed packages:", installed)
                    print(install_logs)
                if installed:
                    result = await self._run_subprocess([python_exe, str(script_path)], workdir)

        reasoning = None
        if result.returncode != 0:
            analysis_prompt = prompts.get_execution_failure_reasoning_prompt(
                code, result.stdout, result.stderr
            )
            reasoning = await query_llm_async(
                analysis_prompt, temperature=LLM_CONFIG["temperature"]["execution"]
            )
            if self.verbose:
                print("CodeExecutorAgent reasoning about failure:\n", reasoning)

        execution = ExecutionResult(
            success=result.returncode == 0,
            stdout=result.stdout,
            stderr=result.stderr,
            error_type=None if result.returncode == 0 else "execution_error",
            packages_installed=packages_installed or None,
            reasoning=reasoning,
        )
        return execution.to_dict()


class HPCAgent(Agent):
    """Submit generated scripts to an HPC scheduler instead of running locally."""

    _DEFAULT_OPTIONS: dict[str, Any] = {
        "scheduler": "pbs",
        "job_name": "agentic_lab_job",
        "account": "GeomicVar",
        "pbs_select": "1:system=sophia",
        "pbs_filesystems": "home:grand",
        "pbs_walltime": "01:00:00",
        "pbs_queue": "by-gpu",
        "modules": [],
        "pre_run_commands": [],
        "submit_command": None,
        "status_poll_interval": 10,
        "status_max_checks": 60,
    }

    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        self.submission_counter = 0

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @staticmethod
    def _python_executable(conda_env_path: str | None) -> str:
        if not conda_env_path:
            return "python"
        candidates = [
            Path(conda_env_path) / "bin" / "python",
            Path(conda_env_path) / "Scripts" / "python.exe",
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return "python"

    def _merge_options(self, overrides: Mapping[str, Any] | None) -> dict[str, Any]:
        merged = dict(self._DEFAULT_OPTIONS)
        if overrides:
            for key, value in overrides.items():
                if value is not None:
                    merged[key] = value
        return merged

    def _render_job_script(
        self,
        *,
        script_path: Path,
        working_dir: Path,
        python_exe: str,
        options: Mapping[str, Any],
        stdout_path: Path,
        stderr_path: Path,
    ) -> str:
        directives: list[str] = ["#!/bin/bash"]
        job_name = options.get("job_name", "agentic_lab_job")
        directives.append(f"#PBS -N {job_name}")
        account = options.get("account")
        if account:
            directives.append(f"#PBS -A {account}")
        select = options.get("pbs_select")
        if select:
            directives.append(f"#PBS -l select={select}")
        filesystems = options.get("pbs_filesystems")
        if filesystems:
            directives.append(f"#PBS -l filesystems={filesystems}")
        walltime = options.get("pbs_walltime")
        if walltime:
            directives.append(f"#PBS -l walltime={walltime}")
        queue = options.get("pbs_queue")
        if queue:
            directives.append(f"#PBS -q {queue}")
        directives.append(f"#PBS -o {stdout_path}")
        directives.append(f"#PBS -e {stderr_path}")

        body: list[str] = ["set -euo pipefail", f"cd {working_dir}"]
        modules: Sequence[str] = options.get("modules", []) or []
        for module in modules:
            body.append(f"module load {module}")

        for command in options.get("pre_run_commands", []) or []:
            body.append(command)

        body.append(f"{python_exe} {script_path}")
        return "\n".join(directives + ["", *body]) + "\n"

    def _write_job_script(
        self,
        *,
        script_contents: str,
        working_dir: Path,
        iteration: int,
    ) -> Path:
        job_dir = working_dir / "hpc_jobs"
        job_dir.mkdir(exist_ok=True)
        job_script = job_dir / f"iteration_{iteration:02d}_{self.submission_counter:02d}.sh"
        job_script.write_text(script_contents)
        return job_script

    async def _run_subprocess(self, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                command,
                cwd=str(cwd),
                capture_output=True,
                text=True,
            ),
        )

    @staticmethod
    def _submit_command(job_script: Path, options: Mapping[str, Any]) -> list[str]:
        custom = options.get("submit_command")
        if custom:
            return [str(arg) for arg in custom]

        scheduler = (options.get("scheduler") or "pbs").lower()
        if scheduler == "pbs":
            return ["qsub", str(job_script)]
        raise ValueError(f"Unsupported scheduler '{scheduler}'. Provide submit_command override.")

    @staticmethod
    def _extract_job_id(stdout: str, stderr: str) -> str | None:
        text = "\n".join(filter(None, [stdout, stderr]))
        patterns = [
            re.compile(r"Submitted batch job (\S+)", re.IGNORECASE),
            re.compile(r"JobID[:\s]+(\S+)", re.IGNORECASE),
            re.compile(r"submitted as job (\S+)", re.IGNORECASE),
            re.compile(r"^(\d+\.\S+)$", re.IGNORECASE),
            re.compile(r"^(\d+)$", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(text)
            if match:
                return match.group(1)
        return None

    @staticmethod
    def _status_command(job_id: str, options: Mapping[str, Any]) -> list[str] | None:
        scheduler = (options.get("scheduler") or "pbs").lower()
        if scheduler == "pbs":
            return ["qstat", job_id]
        return None

    @staticmethod
    def _job_still_listed(
        job_id: str,
        scheduler: str,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> bool:
        scheduler = scheduler.lower()
        text = "\n".join(filter(None, [stdout, stderr]))
        if scheduler == "pbs":
            if returncode != 0:
                return False
            if job_id in text:
                return True
            return bool(text.strip())
        return bool(text.strip())

    async def _poll_job(
        self,
        *,
        job_id: str,
        working_dir: Path,
        options: Mapping[str, Any],
    ) -> str | None:
        status_cmd = self._status_command(job_id, options)
        if not status_cmd:
            if self.verbose:
                print("HPCAgent: no status command configured; skipping polling.")
            return None

        interval = int(options.get("status_poll_interval", 10))
        max_checks = int(options.get("status_max_checks", 60))
        scheduler = (options.get("scheduler") or "pbs").lower()

        for check in range(1, max_checks + 1):
            result = await self._run_subprocess(status_cmd, working_dir)
            stdout = result.stdout.strip()
            stderr = result.stderr.strip()
            if self.verbose:
                print(
                    f"HPCAgent status check {check}/{max_checks} for job {job_id} (scheduler={scheduler}):"
                )
                display_stdout = stdout or "(no stdout from status command)"
                if len(display_stdout) > 2000:
                    display_stdout = display_stdout[:2000] + "... [truncated]"
                print(display_stdout)
                if stderr:
                    display_stderr = stderr
                    if len(display_stderr) > 2000:
                        display_stderr = display_stderr[:2000] + "... [truncated]"
                    print("[stderr]", display_stderr)

            still_running = self._job_still_listed(job_id, scheduler, stdout, stderr, result.returncode)
            if not still_running:
                if self.verbose:
                    print(f"HPCAgent: job {job_id} no longer appears in queue output.")
                return "completed"

            await asyncio.sleep(max(interval, 1))

        if self.verbose:
            print(
                f"HPCAgent: job {job_id} still listed after {max_checks} checks; leaving further monitoring to the user."
            )
        return "timeout"

    @staticmethod
    def _safe_read_file(path: Path, *, max_chars: int = 20000) -> str:
        try:
            data = path.read_text()
        except FileNotFoundError:
            return ""
        if len(data) > max_chars:
            return data[-max_chars:]
        return data

    @staticmethod
    def _logs_suggest_success(stdout: str, stderr: str) -> bool:
        if stdout.strip() and not stderr.strip():
            lowered = stdout.lower()
            failure_tokens = ["error", "traceback", "exception", "fail", "segmentation fault"]
            if any(token in lowered for token in failure_tokens):
                return False
            return True
        if not stdout and not stderr:
            return False
        combined = f"{stdout}\n{stderr}".lower()
        failure_tokens = ["error", "traceback", "exception", "fail", "segmentation fault"]
        return not any(token in combined for token in failure_tokens)

    @staticmethod
    def _format_job_details(job_id: str | None, job_state: str | None, exit_status: int | None) -> str:
        details: list[str] = []
        if job_id:
            details.append(f"job id: {job_id}")
        if job_state:
            details.append(f"state: {job_state}")
        if exit_status is not None:
            details.append(f"exit_status: {exit_status}")
        return ", ".join(details) if details else "job metadata unavailable"

    async def _fetch_job_metadata(
        self, job_id: str, working_dir: Path
    ) -> tuple[str | None, int | None, str, str]:
        cmd = ["qstat", "-fx", job_id]
        result = await self._run_subprocess(cmd, working_dir)
        if result.returncode != 0:
            return None, None, result.stdout.strip(), result.stderr.strip()
        text = result.stdout
        job_state = None
        match_state = re.search(r"job_state\s*=\s*(\w+)", text)
        if match_state:
            job_state = match_state.group(1)
        exit_status = None
        match_exit = re.search(r"exit_status\s*=\s*(-?\d+)", text)
        if not match_exit:
            match_exit = re.search(r"Exit_status\s*=\s*(-?\d+)", text)
        if match_exit:
            try:
                exit_status = int(match_exit.group(1))
            except ValueError:
                exit_status = None
        return job_state, exit_status, result.stdout.strip(), result.stderr.strip()

    async def _analyze_failure(self, code: str | None, stdout: str, stderr: str) -> str:
        source = code or "# Code unavailable for analysis."
        prompt = prompts.get_execution_failure_reasoning_prompt(source, stdout, stderr)
        return await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["execution"])

    @action
    async def submit_job(
        self,
        *,
        script_path: str,
        working_directory: str,
        iteration: int,
        code: str | None = None,
        conda_env_path: str | None = None,
        hpc_options: Mapping[str, Any] | None = None,
    ) -> dict:
        self.submission_counter += 1
        workdir = Path(working_directory)
        workdir.mkdir(parents=True, exist_ok=True)
        python_script = Path(script_path)
        if not python_script.exists():
            raise FileNotFoundError(f"Python script not found for HPC submission: {python_script}")
        options = self._merge_options(hpc_options)
        python_exe = self._python_executable(conda_env_path)
        log_basename = f"hpc_job_iter{iteration:02d}_{self.submission_counter:02d}"
        stdout_path = workdir / f"{log_basename}.out"
        stderr_path = workdir / f"{log_basename}.err"
        for path in (stdout_path, stderr_path):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                path.unlink()
        script_contents = self._render_job_script(
            script_path=python_script,
            working_dir=workdir,
            python_exe=python_exe,
            options=options,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
        job_script = self._write_job_script(
            script_contents=script_contents,
            working_dir=workdir,
            iteration=iteration,
        )

        submit_cmd = self._submit_command(job_script, options)
        if self.verbose:
            print("HPCAgent submitting job with command:", " ".join(submit_cmd))

        result = await self._run_subprocess(submit_cmd, workdir)
        job_id = self._extract_job_id(result.stdout, result.stderr)
        if self.verbose:
            if job_id:
                print(f"HPCAgent submitted job {job_id}")
            else:
                print("HPCAgent submission output:", result.stdout.strip())

        submission_ok = result.returncode == 0
        final_status: str | None = None
        if submission_ok and job_id:
            final_status = await self._poll_job(job_id=job_id, working_dir=workdir, options=options)
        log_stdout = ""
        log_stderr = ""
        job_state: str | None = None
        exit_status: int | None = None
        metadata_stdout = ""
        metadata_stderr = ""

        if submission_ok and final_status == "completed":
            log_stdout = self._safe_read_file(stdout_path)
            log_stderr = self._safe_read_file(stderr_path)
            if job_id:
                job_state, exit_status, metadata_stdout, metadata_stderr = await self._fetch_job_metadata(
                    job_id, workdir
                )

        if not log_stdout:
            log_stdout = metadata_stdout or result.stdout
        if not log_stderr:
            log_stderr = metadata_stderr or result.stderr

        job_success = False
        if submission_ok and final_status == "completed":
            if exit_status is not None:
                job_success = exit_status == 0
            else:
                job_success = self._logs_suggest_success(log_stdout, log_stderr)

        details = self._format_job_details(job_id, job_state, exit_status)

        if not submission_ok:
            reasoning = "HPC submission failed; see stderr for details"
            error_type = "hpc_submission_failed"
        elif final_status == "timeout":
            reasoning = f"{details}; monitoring window ended while job remained in the queue"
            error_type = "hpc_submission_pending"
        elif final_status != "completed":
            reasoning = f"{details}; job status could not be confirmed"
            error_type = "hpc_submission_pending"
        elif job_success:
            reasoning = f"HPC job completed successfully ({details})."
            error_type = "hpc_job_succeeded"
        else:
            failure_analysis = await self._analyze_failure(code, log_stdout, log_stderr)
            reasoning = (
                f"HPC job completed but reported a failure ({details}).\n"
                f"Failure analysis:\n{failure_analysis.strip()}"
            )
            error_type = "hpc_job_failed"

        execution = ExecutionResult(
            success=job_success if (submission_ok and final_status == "completed") else False,
            stdout=log_stdout,
            stderr=log_stderr,
            error_type=error_type,
            packages_installed=None,
            reasoning=reasoning,
            job_id=job_id,
        )
        return execution.to_dict()


class CodeReviewerAgent(Agent):
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @action
    async def review_code(self, code: str, execution_result: str) -> str:
        if self.verbose:
            print("CodeReviewerAgent: reviewing code execution results")
        analysis_prompt = prompts.get_code_reviewer_analysis_prompt(code, execution_result)
        analysis = await query_llm_async(analysis_prompt, temperature=LLM_CONFIG["temperature"]["review"])
        fix_prompt = prompts.get_code_reviewer_fix_prompt(code, execution_result, analysis)
        return await query_llm_async(fix_prompt, temperature=LLM_CONFIG["temperature"]["review"])


class CriticAgent(Agent):
    def __init__(self, *, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose

    @action
    async def set_verbose(self, verbose: bool) -> None:
        self.verbose = verbose

    @action
    async def review_iteration(
        self,
        report: str | None,
        code: str | None,
        execution_result: str | None,
        sources: str,
        execution_reasoning: str | None = None,
    ) -> dict:
        report_feedback = None
        code_feedback = None

        if report:
            prompt = prompts.get_document_critique_prompt(report, sources)
            report_feedback = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["critic"])

        if code and execution_result is not None:
            prompt = prompts.get_code_execution_review_prompt(
                code, execution_result, execution_reasoning
            )
            code_feedback = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["critic"])

        summary = None
        if report_feedback or code_feedback:
            prompt = prompts.get_summary_feedback_prompt(report_feedback or "", code_feedback or "")
            summary = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["critic"])

        if self.verbose:
            print("CriticAgent summary:\n", (summary or "No feedback"))

        bundle = CritiqueBundle(
            document_feedback=report_feedback,
            code_feedback=code_feedback,
            summary=summary,
            executor_feedback=execution_reasoning,
        )
        return bundle.to_dict()
