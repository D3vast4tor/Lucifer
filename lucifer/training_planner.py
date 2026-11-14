"""Bounded loaded-model advice; controller rules and truth checks stay authoritative."""
import json

from .training_contract import master_prompt, prompt_receipt


async def plan_meditation(service, ident, bundle):
    run = service.store.get('meditation_runs', ident)
    if run.get('request', {}).get('dry_run'):
        return bundle
    authorities = sorted({row.get('source_authority', 'verified_example') for row in bundle['examples']})
    previous = run.get('teaching_plan')
    if previous and previous.get('authorities') == authorities:
        plan = previous
    else:
        current = service.manager.current()
        default = sorted(authorities, key=lambda value: (
            value not in {'user_correction', 'document_provided_solution', 'reviewed_public_final'}, value))
        plan = {'master_prompt': prompt_receipt('meditation'), 'authorities': authorities,
                'priority': default, 'selection': 'controller_verified_fallback',
                'training_mode': run['request'].get('mode', 'supervised'),
                'oracle_or_admission_data_shared': False, 'provider_reasoning_retained': False}
        if current and current.get('backend') != 'demo' and authorities:
            messages = [{'role': 'system', 'content': master_prompt('meditation')},
                        {'role': 'user', 'content': 'Choose a priority ordering for the available teaching source authorities. '
                         'Return only JSON {"priority":[every available authority exactly once]}. '
                         'This orders TRAINING data only. Do not invent targets, verifier rules or claims of truth. '
                         f'Mode: {plan["training_mode"]}. Available authorities: {json.dumps(authorities)}.'}]
            count = await service.learning._evaluation_token_count(messages, current['id'], 256)
            response = await service.learning._evaluation_answer(messages, None, current['id'], 256, count)
            try:
                value = json.loads(response['answer']) if response.get('complete') else {}
                priority = value.get('priority')
                if (isinstance(priority, list) and all(isinstance(item, str) for item in priority)
                        and len(priority) == len(authorities) and set(priority) == set(authorities)):
                    plan.update(priority=priority, selection='loaded_model_checked_enum_proposal',
                                teacher_model_id=current['id'])
            except (ValueError, TypeError):
                pass
        service.store.update('meditation_runs', ident, teaching_plan=plan)
    # Pure ordering cannot certify a source as true, touch evaluation partitions
    # or change the controller's optimizer bounds/admission thresholds.
    rank = {value: index for index, value in enumerate(plan['priority'])}
    result = dict(bundle)
    result['examples'] = sorted(bundle['examples'], key=lambda row: rank[row.get('source_authority', 'verified_example')])
    return result
