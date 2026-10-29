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
You choose a job; a shared deterministic controller navigates and interacts for you.
A prep job adds ONE ingredient; a delivery job gets a plate, collects soup, and delivers.
Both agents can do either job. Job commitment lasts until completion, target change or timeout.
Use current observations and partner information to avoid redundant work. Partner information
is a prior, not a command: consider current evidence. Do not assume unobserved dynamic states.
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
    if not isinstance(obj.get('reason'), str) or len(obj['reason']) > 1200:
        raise ModelError('Missing or excessively long reason')
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

    def choose(self, context, seed, audit):
        payload = {'model': self.model,
                   'messages': [{'role': 'system', 'content': SYSTEM},
                                {'role': 'user', 'content': json.dumps(context)}],
                   'max_tokens': self.max_tokens, 'temperature': self.temperature,
                   'seed': seed, 'response_format': {'type': 'json_object'},
                   'chat_template_kwargs': {'enable_thinking': False}}
        # Raw REST body: chat_template_kwargs is top-level, not nested in extra_body.
        for attempt in range(self.retries+1):
            start = time.monotonic()
            record = {'attempt': attempt, 'request': payload}
            try:
                response = self._request('chat/completions', payload)
                record['response'] = response
                choice = response['choices'][0]
                if choice.get('finish_reason') != 'stop':
                    raise ModelError(f'Generation did not finish normally: {choice.get("finish_reason")}')
                parsed = parse_choice(choice['message'].get('content'), context['candidates'])
                record['parsed'] = parsed
            except (ModelError, KeyError, IndexError, TypeError) as e:
                record['error'] = str(e)
                record['latency_s'] = time.monotonic()-start
                audit(record)
                if attempt == self.retries:
                    raise ModelError(str(e)) from e
            else:
                record['latency_s'] = time.monotonic()-start
                audit(record)
                return parsed
        raise AssertionError('unreachable')
