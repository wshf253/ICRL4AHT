"""OpenAI-compatible HTTP client, independent of vLLM/Torch installations."""
import json
import os
import re
import time
from urllib.request import Request, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError

SYSTEM = '''You coordinate with one partner in a cooking task. Maximize correct soup deliveries
within the time budget. The public recipe is three units of ingredient 0. A pot cooks
automatically after its third ingredient. Collect cooked soup with a plate, then deliver.
You choose a job; a shared controller navigates and interacts for you.
A prep job adds ONE ingredient; a delivery job gets a plate, collects soup, and delivers.
Both agents can do either job. Job commitment lasts until completion, target change or timeout.
Use current observations and partner information to avoid redundant work. Partner information
is a prior, not a command: consider current evidence. Do not assume unobserved dynamic states.
State encoding: holding=empty means empty hands; empty_plate is a plate without soup;
cooked_soup_on_plate is one plated soup, not two separately held objects.
Agents carry one ingredient OR one plate OR one plated soup at a time.
ingredient_counts maps ingredient names to quantities: {"ingredient_0": 0} means none,
{"ingredient_0": 1} means one unit of ingredient_0, never ingredient_1.
Pot status is empty, filling, cooking, or ready. Only ready soup can be collected.
remaining_cook_steps is a countdown, not a signal to start cooking manually.
Return ONLY one JSON object with keys:
{"task_id": "one of the supplied candidate ids",
 "partner_prediction": "prep|delivery|inspect|recovery|wait",
 "reason": "one short sentence explaining the work allocation"}.
Predict the category of work the partner is pursuing now. Never invent a task id.'''


SYSTEM_MULTI_RECIPE = '''You coordinate with one partner in a cooking task. Maximize correct soup deliveries
within the time budget. observation.recipe gives the current recipe as ingredient counts:
{"ingredient_0": 2, "ingredient_1": 1} means a correct soup has exactly two ingredient_0 and one
ingredient_1. The recipe is fixed for the whole episode. If recipe.source is "unknown", the
recipe has not been observed yet; inspect a recipe indicator before preparing ingredients.
A pot cooks automatically after its third ingredient of ANY type. A pot whose count of some
ingredient exceeds the recipe can never become a correct soup. Delivering a soup that does not
match the recipe costs the same reward a correct delivery earns. A discard job plates a pot's soup
and puts it on a counter instead, which frees the pot at no penalty. Discard jobs are offered for
every cooking or ready pot; discarding a soup that matches the recipe throws away a correct soup.
You choose a job; a shared controller navigates and interacts for you.
A prep job fetches ONE unit of the named ingredient and adds it to the named pot.
A delivery job gets a plate, collects that pot's soup when ready, and delivers it.
Both agents can do either job. Job commitment lasts until completion, target change or timeout.
Use current observations and partner information to avoid redundant work. Partner information
is a prior, not a command: consider current evidence. Do not assume unobserved dynamic states.
State encoding: holding=empty means empty hands; empty_plate is a plate without soup;
cooked_soup_on_plate is one plated soup, not two separately held objects.
Agents carry one ingredient OR one plate OR one plated soup at a time.
ingredient_counts maps ingredient names to quantities: {"ingredient_0": 1, "ingredient_1": 0}
means one unit of ingredient_0 and none of ingredient_1.
Pot status is empty, filling, cooking, or ready. Only ready soup can be collected.
remaining_cook_steps is a countdown, not a signal to start cooking manually.
Return ONLY one JSON object with keys:
{"task_id": "one of the supplied candidate ids",
 "partner_prediction": "prep|delivery|inspect|recovery|wait",
 "reason": "one short sentence explaining the work allocation"}.
Predict the category of work the partner is pursuing now. Never invent a task id.'''


class ModelError(RuntimeError):
    pass


def parse_choice(text, menu):
    if not isinstance(text, str) or not text.strip():
        raise ModelError('Empty final content')
    text = text.strip()
    if text.startswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
    try:
        obj = json.loads(text)
    except (ValueError, TypeError) as e:
        raise ModelError('Final content is not a JSON object') from e
    if not isinstance(obj, dict) or obj.get('task_id') not in {c['id'] for c in menu}:
        raise ModelError('Invalid task_id')
    if obj.get('partner_prediction') not in ('prep', 'delivery', 'inspect', 'recovery', 'wait'):
        raise ModelError('Invalid partner_prediction')
    # Generation is bounded by max_tokens; preserve long reasons for auditing.
    if not isinstance(obj.get('reason'), str):
        raise ModelError('Missing or non-string reason')
    return obj


class ChatClient:
    def __init__(self, base_url, model='qwen35-27b', max_tokens=512, timeout=120,
                 temperature=0.0, retries=1):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.temperature = temperature
        self.retries = retries
        self.key = os.environ.get('ICRL_LLM_API_KEY', '')
        # Cluster loopback/node addresses should not be routed through HTTP proxies.
        self.opener = build_opener(ProxyHandler({}))

    def _request(self, endpoint, payload=None):
        headers = {'Content-Type': 'application/json'}
        if self.key:
            headers['Authorization'] = 'Bearer '+self.key
        req = Request(self.base_url+'/'+endpoint, headers=headers,
                      data=None if payload is None else json.dumps(payload).encode())
        try:
            with self.opener.open(req, timeout=self.timeout) as response:
                return json.load(response)
        except HTTPError as e:
            detail = e.read(2000).decode(errors='replace')
            raise ModelError(f'HTTP {e.code}: {detail}') from e
        except (URLError, TimeoutError, ValueError) as e:
            raise ModelError(f'Model request failed: {type(e).__name__}: {e}') from e

    def check_model(self):
        ids = [m['id'] for m in self._request('models').get('data', [])]
        if self.model not in ids:
            raise ModelError(f'Model {self.model!r} is not served; available IDs: {ids}')
        return ids

    def choose(self, context, seed, audit, system=None):
        messages = [{'role': 'system', 'content': system or SYSTEM},
                    {'role': 'user', 'content': json.dumps(context)}]
        # Raw REST body: chat_template_kwargs is top-level, not nested in extra_body.
        for attempt in range(self.retries+1):
            payload = {'model': self.model, 'messages': messages,
                       'max_tokens': self.max_tokens, 'temperature': self.temperature,
                       'seed': seed, 'response_format': {'type': 'json_object'},
                       'chat_template_kwargs': {'enable_thinking': False}}
            start = time.monotonic()
            record = {'attempt': attempt, 'request': payload, 'feedback_retry': attempt > 0 and len(messages) > 2}
            content = None
            try:
                response = self._request('chat/completions', payload)
                record['response'] = response
                choice = response['choices'][0]
                content = choice['message'].get('content')
                if choice.get('finish_reason') != 'stop':
                    raise ModelError(f'Generation did not finish normally: {choice.get("finish_reason")}')
                parsed = parse_choice(content, context['candidates'])
                record['parsed'] = parsed
            except (ModelError, KeyError, IndexError, TypeError) as e:
                record['error'] = str(e)
                record['latency_s'] = time.monotonic()-start
                audit(record)
                if attempt == self.retries:
                    raise ModelError(str(e)) from e
                if isinstance(content, str) and content.strip():
                    # Resending the identical request at temperature 0 returns the
                    # identical answer, so a plain retry cannot recover. Show the
                    # rejected answer and the reason once; every attempt is logged.
                    ids = [c['id'] for c in context['candidates']]
                    messages = messages[:2] + [
                        {'role': 'assistant', 'content': content},
                        {'role': 'user', 'content': f'Your answer was rejected: {e}. task_id must be exactly '
                                                    f'one of {json.dumps(ids)}. Return only the JSON object.'}]
            else:
                record['latency_s'] = time.monotonic()-start
                audit(record)
                return parsed
        raise AssertionError('unreachable')
