"""Academy agent implementations for the Agentic Lab workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

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
    ) -> dict:
        report_feedback = None
        code_feedback = None

        if report:
            prompt = prompts.get_document_critique_prompt(report, sources)
            report_feedback = await query_llm_async(prompt, temperature=LLM_CONFIG["temperature"]["critic"])

        if code and execution_result is not None:
            prompt = prompts.get_code_execution_review_prompt(code, execution_result)
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
        )
        return bundle.to_dict()
