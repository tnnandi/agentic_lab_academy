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
) -> None:
    
    def _format_pi_changes(feedback: CritiqueBundle | None) -> str | None:
        """Summarize critic remarks so the PI can adjust the plan next round."""
        if not feedback:
            return None

        sections: list[str] = []
        if feedback.summary:
            sections.append(f"Overall summary from critic:\n{feedback.summary}")
        if feedback.document_feedback:
            sections.append(f"Document feedback:\n{feedback.document_feedback}")
        if feedback.code_feedback:
            sections.append(f"Code feedback:\n{feedback.code_feedback}")
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

    workdir = Path.cwd() / "workspace_runs"
    workdir.mkdir(exist_ok=True)

    if verbose:
        print("Workspace for generated scripts:", workdir)

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

        await asyncio.gather(
            pi.configure(verbose=verbose, max_rounds=MAX_ROUNDS),
            browsing.set_verbose(verbose),
            research.set_verbose(verbose),
            code_writer.set_verbose(verbose),
            code_executor.set_verbose(verbose),
            code_reviewer.set_verbose(verbose),
            critic.set_verbose(verbose),
        )

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

        while True:
            decision = (await _to_thread(input, "PI: Do you want to proceed with the plan? (y/n): ")).strip().lower()
            if decision == "y":
                print("PI: User agreed to the plan.")
                break
            if decision == "n":
                changes = await _to_thread(input, "PI: Please input the suggested changes: ")
                plan_dict = await pi.create_plan(sources=sources, topic=topic, mode=mode, changes=changes)
                plan = PlanResult.from_dict(plan_dict)
                continue
            print("PI: Invalid input. Please enter 'y' or 'n'.")

        research_result: ResearchArtifact | None = None
        code_artifact: CodeArtifact | None = None
        execution_result: ExecutionResult | None = None
        critic_feedback: CritiqueBundle | None = None

        # 
        for iteration in range(MAX_ROUNDS):
            print("=" * 80)
            print(f"Iteration {iteration + 1}/{MAX_ROUNDS}")
            print("=" * 80)

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

            if mode in {"code_only", "both"}:
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
                    feedback = critic_feedback.code_feedback if critic_feedback else ""
                    code_dict = await code_writer.improve_code(
                        code=code_artifact.code,
                        feedback=feedback or "",
                        iteration=iteration,
                    )
                code_artifact = CodeArtifact.from_dict(code_dict)

                execution_result: ExecutionResult | None = None
                execution_transcript = ""

                for attempt in range(1, MAX_EXECUTION_ATTEMPTS + 1):
                    exec_dict = await code_executor.execute_code(
                        code=code_artifact.code,
                        working_directory=str(workdir),
                        iteration=iteration,
                        conda_env_path=conda_env,
                    )
                    execution_result = ExecutionResult.from_dict(exec_dict)

                    execution_transcript = (
                        f"SUCCESS: {execution_result.success}\n"
                        f"STDOUT:\n{execution_result.stdout}\n\n"
                        f"STDERR:\n{execution_result.stderr}\n\n"
                        f"PACKAGES_INSTALLED: {execution_result.packages_installed or []}\n"
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

                    # set_trace()

                if execution_result and not execution_result.success:
                    review = await code_reviewer.review_code(code_artifact.code, execution_transcript)
                    improved_code = utils.extract_code_only(review)
                    if improved_code and improved_code != code_artifact.code:
                        code_artifact = CodeArtifact(code=improved_code, iteration=iteration)

            else:
                execution_transcript = None

            critic_dict = await critic.review_iteration(
                report=research_result.content if research_result else None,
                code=code_artifact.code if code_artifact else None,
                execution_result=execution_transcript,
                sources=sources,
            )
            critic_feedback = CritiqueBundle.from_dict(critic_dict)

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

            utils.save_output(
                report=research_result.content if research_result else "",
                code=code_artifact.code if code_artifact else "",
                execution_result=execution_transcript or (execution_result.stdout if execution_result else ""),
                timestamp=timestamp,
                iteration=iteration,
            )

            if execution_result and execution_result.success:
                print("Code executed successfully. Stopping iterations.")
                break

    executor.shutdown(wait=False)
