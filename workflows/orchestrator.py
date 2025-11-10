"""Async workflow orchestration using Academy agents."""
from __future__ import annotations
"""To fix:
1. Allow proper message passing from the critic agent to the PI agent after the successful completion of a single round (though multiple rounds may not be needed right now)
2. Keep a record of all conversations between the agents so that same mistakes aren't done again
3. Create a proper schematic showing all informations flows between user and the agents, and among the agents themselves, as well as points where humans are allowed in the loop
4. 

"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence
from datetime import datetime

from academy.exchange import LocalExchangeFactory
from academy.manager import Manager

from pdb import set_trace

try:
    from ..academy_agents import (
        BrowsingAgent,
        CodeExecutorAgent,
        HPCAgent,
        CodeReviewerAgent,
        CodeWriterAgent,
        CriticAgent,
        PrincipalInvestigatorAgent,
        ResearchAgent,
    )
    from ..config import MAX_ROUNDS
    from ..models import CodeArtifact, CritiqueBundle, ExecutionResult, PlanResult, ResearchArtifact
    from .. import utils
except ImportError:
    from academy_agents import (
        BrowsingAgent,
        CodeExecutorAgent,
        HPCAgent,
        CodeReviewerAgent,
        CodeWriterAgent,
        CriticAgent,
        PrincipalInvestigatorAgent,
        ResearchAgent,
    )
    from config import MAX_ROUNDS
    from models import CodeArtifact, CritiqueBundle, ExecutionResult, PlanResult, ResearchArtifact
    import utils


MAX_EXECUTION_ATTEMPTS = 3 # Number of loops between the code executor and the code writer agent
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


async def _to_thread(func, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, func, *args)


async def run_workflow(
    *,
    topic: str,
    mode: str,
    quick_search: bool,
    pdfs: Sequence[str] | None,
    links: Sequence[str] | None,
    files_dir: str | None,
    conda_env: str | None,
    verbose: bool = True,
    use_hpc: bool = False,
) -> None:
    
    def _format_pi_changes(feedback: CritiqueBundle | None) -> str | None:
        """Summarize critic and executor remarks so the PI can adjust the plan next round."""
        if not feedback:
            return None

        sections: list[str] = []
        if feedback.summary:
            sections.append(f"Overall summary from critic:\n{feedback.summary}")
        if feedback.document_feedback:
            sections.append(f"Document feedback:\n{feedback.document_feedback}")
        if feedback.code_feedback:
            sections.append(f"Code feedback:\n{feedback.code_feedback}")
        executor_notes = getattr(feedback, "executor_feedback", None)
        if executor_notes:
            sections.append(f"Executor diagnostics:\n{executor_notes}")
        return "\n\n".join(sections) if sections else None

    # Read content from the pdfs
    pdf_content = ""
    if pdfs:
        pdf_content = await _to_thread(utils.process_pdfs, list(pdfs))
    # set_trace()

    # Read files from files_dir
    files_dir_content = ""
    if files_dir:
        files_dir_content = await _to_thread(utils.explore_files_directory, files_dir)

    workspace_root = Path.cwd() / "workspace_runs"
    workspace_root.mkdir(exist_ok=True)
    run_dir = workspace_root / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    conversation_log_path = run_dir / f"conversation_log_{timestamp}.jsonl"
    hpc_script_counter = 0

    def _materialize_hpc_script(code: str, iteration_idx: int) -> Path:
        nonlocal hpc_script_counter
        hpc_script_counter += 1
        scripts_dir = run_dir / "generated_code"
        scripts_dir.mkdir(exist_ok=True)
        script_path = scripts_dir / f"hpc_iteration_{iteration_idx:02d}_{hpc_script_counter:02d}.py"
        script_path.write_text(code)
        return script_path

    def _log_event(role: str, message: str, iteration_idx: int | None = None, metadata: dict | None = None) -> None:
        utils.append_conversation_log(
            log_path=conversation_log_path,
            role=role,
            message=message,
            iteration=iteration_idx,
            metadata=metadata or {},
        )

    if verbose:
        print("Workspace for generated scripts:", run_dir)

    executor = ThreadPoolExecutor(max_workers=8)
    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(), executors=executor  #TN: can replace LocalExchangeFactory with ProxyStoreExchangeFactory for connecting to cluster?
    ) as manager:
        pi = await manager.launch(PrincipalInvestigatorAgent)
        browsing = await manager.launch(BrowsingAgent)
        research = await manager.launch(ResearchAgent)
        code_writer = await manager.launch(CodeWriterAgent)
        code_executor = await manager.launch(CodeExecutorAgent)
        code_reviewer = await manager.launch(CodeReviewerAgent)
        critic = await manager.launch(CriticAgent)
        hpc_agent = await manager.launch(HPCAgent) if use_hpc else None

        tasks = [
            pi.configure(verbose=verbose, max_rounds=MAX_ROUNDS),
            browsing.set_verbose(verbose),
            research.set_verbose(verbose),
            code_writer.set_verbose(verbose),
            code_executor.set_verbose(verbose),
            code_reviewer.set_verbose(verbose),
            critic.set_verbose(verbose),
        ]
        if hpc_agent:
            tasks.append(hpc_agent.set_verbose(verbose))
        await asyncio.gather(*tasks)

        if quick_search:
            search_result = await browsing.quick_search(topic)
            print(search_result)
            return

        sources = await browsing.gather_sources(
            topic=topic,
            pdf_content=pdf_content,
            links=list(links) if links else None,
            files_dir_content=files_dir_content,
        )

        # Let the PI create the initial plan
        plan_dict = await pi.create_plan(sources=sources, topic=topic, mode=mode)
        plan = PlanResult.from_dict(plan_dict)
        _log_event(
            "PrincipalInvestigatorAgent",
            "Initial plan created.",
            metadata={"plan": plan.plan, "reasoning": plan.reasoning},
        )

        while True:
            decision = (await _to_thread(input, "PI: Do you want to proceed with the plan? (y/n): ")).strip().lower()
            if decision == "y":
                print("PI: User agreed to the plan.")
                _log_event("User", "Approved plan.", metadata={"decision": decision})
                break
            if decision == "n":
                changes = await _to_thread(input, "PI: Please input the suggested changes: ")
                plan_dict = await pi.create_plan(sources=sources, topic=topic, mode=mode, changes=changes)
                plan = PlanResult.from_dict(plan_dict)
                _log_event(
                    "PrincipalInvestigatorAgent",
                    "Plan updated based on user feedback.",
                    metadata={"plan": plan.plan, "reasoning": plan.reasoning, "user_changes": changes},
                )
                continue
            print("PI: Invalid input. Please enter 'y' or 'n'.")

        research_result: ResearchArtifact | None = None
        code_artifact: CodeArtifact | None = None
        execution_result: ExecutionResult | None = None
        critic_feedback: CritiqueBundle | None = None

        for iteration in range(MAX_ROUNDS):
            print("=" * 80)
            print(f"Iteration {iteration + 1}/{MAX_ROUNDS}")
            print("=" * 80)
            _log_event(
                "Orchestrator",
                "Starting iteration.",
                iteration,
                {"max_rounds": MAX_ROUNDS, "plan_excerpt": plan.plan[:400]},
            )
            executor_reasoning_note = "Code path not executed this iteration."

            if mode in {"research_only", "both"}:
                if iteration == 0 or not research_result:
                    research_dict = await research.draft_document(
                        sources=sources,
                        topic=topic,
                        plan_section=plan.plan,
                        iteration=iteration,
                    )
                else:
                    feedback = critic_feedback.document_feedback if critic_feedback else ""
                    research_dict = await research.improve_document(
                        draft=research_result.content,
                        feedback=feedback or "",
                        iteration=iteration,
                    )
                research_result = ResearchArtifact.from_dict(research_dict)
                _log_event(
                    "ResearchAgent",
                    "Produced research draft.",
                    iteration,
                    {"iteration": research_result.iteration, "excerpt": research_result.content[:600]},
                )

            if mode in {"code_only", "both"}:
                executor_reasoning_note = "Awaiting execution results."
                if iteration == 0 or not code_artifact:
                    coding_plan = await code_writer.create_coding_plan(sources, topic, plan.plan)
                    print("\n" + "=" * 80)
                    print("CodeWriterAgent proposed coding plan:\n")
                    print(coding_plan.strip())
                    print("=" * 80 + "\n")
                    while True:
                        approved = (
                            await _to_thread(input, "CodeWriter: Approve coding plan? (y/n): ")
                        ).strip().lower()
                        if approved == "y":
                            break
                        if approved == "n":
                            feedback = await _to_thread(input, "Provide feedback for coding plan: ")
                            coding_plan = await code_writer.improve_coding_plan(feedback, coding_plan)
                            print("\n" + "=" * 80)
                            print("CodeWriterAgent improved coding plan:\n")
                            print(coding_plan.strip())
                            print("=" * 80 + "\n")
                        else:
                            print("Invalid input. Please respond with y/n.")
                    code_dict = await code_writer.create_code(
                        sources=sources,
                        topic=topic,
                        plan_section=plan.plan,
                        coding_plan=coding_plan,
                        iteration=iteration,
                    )
                else:
                    feedback_sections: list[str] = []
                    if critic_feedback and critic_feedback.executor_feedback:
                        feedback_sections.append(f"Executor diagnostics:\n{critic_feedback.executor_feedback}")
                    if critic_feedback and critic_feedback.code_feedback:
                        feedback_sections.append(critic_feedback.code_feedback)
                    feedback = "\n\n".join(feedback_sections)
                    code_dict = await code_writer.improve_code(
                        code=code_artifact.code,
                        feedback=feedback or "",
                        iteration=iteration,
                    )
                code_artifact = CodeArtifact.from_dict(code_dict)
                _log_event(
                    "CodeWriterAgent",
                    "Produced code artifact.",
                    iteration,
                    {"iteration": code_artifact.iteration, "code_preview": code_artifact.code[:600]},
                )

                execution_result: ExecutionResult | None = None
                execution_transcript = ""

                if use_hpc:
                    if not hpc_agent:
                        raise RuntimeError("HPCAgent not initialized despite --use_hpc flag.")
                    for attempt in range(1, MAX_EXECUTION_ATTEMPTS + 1):
                        script_path = _materialize_hpc_script(code_artifact.code, iteration)
                        exec_dict = await hpc_agent.submit_job(
                            script_path=str(script_path),
                            working_directory=str(run_dir),
                            iteration=iteration,
                            code=code_artifact.code,
                            conda_env_path=conda_env,
                        )
                        execution_result = ExecutionResult.from_dict(exec_dict)
                        executor_reasoning_note = (
                            execution_result.reasoning
                            or "HPC job submitted; awaiting cluster execution results."
                        )
                        execution_transcript = (
                            f"HPC attempt {attempt}: success={execution_result.success}\n"
                            f"JOB_ID: {execution_result.job_id or 'unknown'}\n"
                            f"STDOUT:\n{execution_result.stdout}\n\n"
                            f"STDERR:\n{execution_result.stderr}\n"
                        )
                        _log_event(
                            "HPCAgent",
                            "HPC attempt completed.",
                            iteration,
                            {
                                "attempt": attempt,
                                "job_id": execution_result.job_id,
                                "success": execution_result.success,
                                "reasoning": execution_result.reasoning,
                                "error_type": execution_result.error_type,
                                "stdout": execution_result.stdout[:1000],
                                "stderr": execution_result.stderr[:1000],
                            },
                        )

                        if execution_result.error_type == "hpc_submission_pending":
                            print(
                                "HPCAgent monitoring window ended while the job is still queued; please monitor it manually."
                            )
                            break

                        if execution_result.success:
                            break

                        reasoning_text = execution_result.reasoning or "No automated reasoning available."
                        print("HPCAgent analysis of failure:\n", reasoning_text, "\n")

                        feedback = (
                            f"The HPC execution attempt {attempt}/{MAX_EXECUTION_ATTEMPTS} failed.\n"
                            "Executor analysis:\n"
                            f"{reasoning_text}\n\n"
                            "Execution transcript:\n"
                            f"{execution_transcript}"
                        )

                        improved_dict = await code_writer.improve_code(
                            code=code_artifact.code,
                            feedback=feedback,
                            iteration=iteration,
                        )
                        improved_artifact = CodeArtifact.from_dict(improved_dict)

                        if improved_artifact.code == code_artifact.code:
                            break

                        code_artifact = improved_artifact
                        _log_event(
                            "CodeWriterAgent",
                            "Refined code artifact after HPC feedback.",
                            iteration,
                            {"code_preview": code_artifact.code[:600]},
                        )
                else:
                    for attempt in range(1, MAX_EXECUTION_ATTEMPTS + 1):
                        exec_dict = await code_executor.execute_code(
                            code=code_artifact.code,
                            working_directory=str(run_dir),
                            iteration=iteration,
                            conda_env_path=conda_env,
                        )
                        execution_result = ExecutionResult.from_dict(exec_dict)
                        executor_reasoning_note = (
                            execution_result.reasoning
                            or f"Execution attempt {attempt} "
                            f"{'succeeded' if execution_result.success else 'failed without detailed reasoning.'}"
                        )

                        execution_transcript = (
                            f"SUCCESS: {execution_result.success}\n"
                            f"STDOUT:\n{execution_result.stdout}\n\n"
                            f"STDERR:\n{execution_result.stderr}\n\n"
                            f"PACKAGES_INSTALLED: {execution_result.packages_installed or []}\n"
                        )
                        _log_event(
                            "CodeExecutorAgent",
                            "Execution attempt completed.",
                            iteration,
                            {
                                "attempt": attempt,
                                "success": execution_result.success,
                                "reasoning": execution_result.reasoning,
                                "stdout": execution_result.stdout[:1000],
                                "stderr": execution_result.stderr[:1000],
                            },
                        )

                        if execution_result.success:
                            break

                        reasoning_text = execution_result.reasoning or "No automated reasoning available."
                        print("CodeExecutorAgent analysis of failure:\n", reasoning_text, "\n")

                        feedback = (
                            f"The execution attempt {attempt}/{MAX_EXECUTION_ATTEMPTS} failed.\n"
                            "Executor analysis:\n"
                            f"{reasoning_text}\n\n"
                            "Execution transcript:\n"
                            f"{execution_transcript}"
                        )

                        improved_dict = await code_writer.improve_code(
                            code=code_artifact.code,
                            feedback=feedback,
                            iteration=iteration,
                        )
                        improved_artifact = CodeArtifact.from_dict(improved_dict)

                        if improved_artifact.code == code_artifact.code:
                            # No progress from code writer; rely on reviewer fallback below.
                            break

                        code_artifact = improved_artifact
                        _log_event(
                            "CodeWriterAgent",
                            "Refined code artifact after executor feedback.",
                            iteration,
                            {"code_preview": code_artifact.code[:600]},
                        )

                        # set_trace()

                if execution_result and not execution_result.success:
                    allow_reviewer = (
                        not use_hpc
                        or execution_result.error_type in {"hpc_job_failed", "hpc_submission_failed"}
                    )
                    if allow_reviewer:
                        review = await code_reviewer.review_code(code_artifact.code, execution_transcript)
                        improved_code = utils.extract_code_only(review)
                        if improved_code and improved_code != code_artifact.code:
                            code_artifact = CodeArtifact(code=improved_code, iteration=iteration)
                            _log_event(
                                "CodeReviewerAgent",
                                "Reviewer adjusted code after failed execution.",
                                iteration,
                                {"code_preview": code_artifact.code[:600]},
                            )

            else:
                execution_transcript = None
                executor_reasoning_note = "Code path skipped due to selected mode."

            critic_dict = await critic.review_iteration(
                report=research_result.content if research_result else None,
                code=code_artifact.code if code_artifact else None,
                execution_result=execution_transcript,
                execution_reasoning=executor_reasoning_note,
                sources=sources,
            )
            critic_feedback = CritiqueBundle.from_dict(critic_dict)
            _log_event(
                "CriticAgent",
                "Provided iteration critique.",
                iteration,
                {
                    "document_feedback": critic_feedback.document_feedback,
                    "code_feedback": critic_feedback.code_feedback,
                    "summary": critic_feedback.summary,
                    "executor_feedback": getattr(critic_feedback, "executor_feedback", None),
                },
            )

            # Refresh the PI’s plan for the next iteration using the latest critic feedback.
            if iteration + 1 < MAX_ROUNDS:
                plan_changes = _format_pi_changes(critic_feedback)
                if plan_changes:
                    plan_dict = await pi.create_plan(
                        sources=sources,
                        topic=topic,
                        mode=mode,
                        changes=plan_changes,
                    )
                    plan = PlanResult.from_dict(plan_dict)
                    _log_event(
                        "PrincipalInvestigatorAgent",
                        "Updated plan after critic/executor feedback.",
                        iteration,
                        {"plan": plan.plan, "changes": plan_changes},
                    )

            utils.save_output(
                report=research_result.content if research_result else "",
                code=code_artifact.code if code_artifact else "",
                execution_result=execution_transcript or (execution_result.stdout if execution_result else ""),
                timestamp=timestamp,
                iteration=iteration,
            )
            _log_event(
                "Orchestrator",
                "Saved iteration artifacts.",
                iteration,
                {"timestamp": timestamp},
            )

            if execution_result and execution_result.success:
                print("Code executed successfully. Stopping iterations.")
                break

            if use_hpc and execution_result and execution_result.error_type == "hpc_submission_pending":
                print("HPC job is still queued or monitoring timed out; please watch the cluster queue.")
                break

    executor.shutdown(wait=False)
