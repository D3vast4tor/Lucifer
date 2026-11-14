"""Human-readable training outcomes; absent measurements stay unknown."""


LEGEND = (
    'HOW TO READ THIS\n\n'
    'Solved tasks: verified final answers, passing hidden tests or correct final state.\n\n'
    'Baseline → candidate: same tasks, tools and budgets before and after training.\n\n'
    'Gains / regressions: newly solved tasks / previously solved tasks now failed.\n\n'
    'Executed updates include work later rolled back. Retained updates belong to the selected checkpoint.\n\n'
    'Loss ↓: better target-token prediction. KL: output-distribution drift. Neither is an IQ score.\n\n'
    'Unknown / incomplete: missing evidence, never a pass.\n\n'
    'Contract failures: a completed response did not follow the task answer/tool format; counted as failed tasks.\n\n'
    'A saved candidate is not active. Activation is explicit; restoration must be verified.'
)
LABELS = {'novel_reasoning': 'Novel reasoning', 'coding': 'Code repair',
          'agent_tasks': 'Agent tasks', 'recovery_transfer': 'Recovery / transfer'}


def outcome_lines(job):
    job = job or {}
    report = job.get('behavioral_report') or (job.get('admission_report') or {}).get('behavioral')
    lines = ['BEHAVIORAL OUTCOMES']
    if not report:
        return lines + ['Not measured yet. Training loss is not a task-success result.']
    if report.get('diagnostic_recheck'):
        lines.append('ASSESSMENT // saved-artifact verifier correction; original results preserved')
    lines.append('BASELINE → CANDIDATE // solved / total')
    for row in report.get('categories', []):
        label = LABELS.get(row.get('category'), str(row.get('category', 'Unknown')))
        lines.append(f"{label}: {row.get('baseline_solved', '?')}/{row.get('total', '?')} → "
                     f"{row.get('candidate_solved', '?')}/{row.get('total', '?')}")
    lines.append(f"GAINS / REGRESSIONS // {report.get('gains', '?')} / {report.get('regressions', '?')}")
    lines.append('EVIDENCE // ' + ('complete' if report.get('complete') is True else 'incomplete — cannot admit'))
    lines.append('OUTCOME // ' + str(report.get('quality_status', 'unknown')).replace('_', ' ').upper())
    lines.append(f"CONTRACT FAILURES // {report.get('baseline_contract_failures', '?')} → {report.get('candidate_contract_failures', '?')}")
    lines.append(f"TOOL ERRORS / EXHAUSTED BUDGETS // {report.get('candidate_tool_errors', '?')} / {report.get('candidate_budget_exhausted', '?')}")
    paired = report.get('paired_test') or {}
    if 'p_value' in paired:
        lines.append(f"IMPROVEMENT EVIDENCE // p={paired['p_value']:.4g} / requires ≤ {paired['alpha']:.4g}")
    sequential = report.get('sequential_gate') or (job.get('admission_report') or {}).get('sequential_gate') or {}
    if sequential and sequential.get('alpha') != paired.get('alpha'):
        lines.append(f"RUN CONFIDENCE LIMIT // attempt {sequential['attempt']} / requires p ≤ {sequential['alpha']:.4g}")
    return lines


def retained_updates(job):
    explicit = job.get('retained_checkpoint_updates')
    if type(explicit) is int and explicit >= 0:
        return explicit
    receipt = job.get('weight_update_receipt') or {}
    if receipt.get('checkpoint_path') and receipt.get('checkpoint_path') == job.get('checkpoint_path'):
        value = receipt.get('successful_updates')
        if type(value) is int and value >= 0:
            return value
    return None
