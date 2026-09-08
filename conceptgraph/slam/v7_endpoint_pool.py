"""At most three active HTTP calls, with sequential timeout failover per request."""
from contextlib import contextmanager
import threading
import time
from urllib.parse import urlsplit
import httpx


def validate_urls(urls):
    urls = list(dict.fromkeys(str(u).rstrip('/') for u in urls))
    if not 1 <= len(urls) <= 3:
        raise ValueError('Provide one to three distinct VLM endpoints')
    for url in urls:
        parsed = urlsplit(url)
        if parsed.scheme not in {'http', 'https'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('VLM endpoint must be an HTTP(S) base URL without credentials/query')
    return urls


class EndpointPool:
    def __init__(self, urls, timeout, max_timeout_retries=3):
        self.urls = validate_urls(urls)
        self.timeout = float(timeout)
        self.max_timeout_retries = int(max_timeout_retries)
        if self.timeout <= 0 or self.max_timeout_retries != 3:
            raise ValueError('Positive timeout and exactly three timeout retries required')
        self.condition = threading.Condition()
        self.busy = set()
        self.timeouts = dict.fromkeys(self.urls, 0)
        self.assignments = dict.fromkeys(self.urls, 0)

    @contextmanager
    def lease(self, previous=None):
        with self.condition:
            while True:
                eligible = [u for u in self.urls if len(self.urls) == 1 or u != previous]
                free = [u for u in eligible if u not in self.busy]
                if free:
                    url = min(free, key=lambda u: (self.timeouts[u], self.assignments[u], self.urls.index(u)))
                    self.busy.add(url)
                    self.assignments[url] += 1
                    break
                self.condition.wait()
        try:
            yield url
        finally:
            with self.condition:
                self.busy.remove(url)
                self.condition.notify_all()

    def request(self, payload, record_attempt):
        previous = None
        attempts = []
        for number in range(1, self.max_timeout_retries + 2):
            with self.lease(previous) as url:
                started = time.perf_counter()
                response = None
                error = None
                timed_out = False
                try:
                    with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                        response = client.post(url + '/api/chat', json=payload)
                    if response.status_code in (408, 504):
                        timed_out = True
                        error = f'HTTP {response.status_code} timeout'
                    else:
                        response.raise_for_status()
                except httpx.TimeoutException as exc:
                    timed_out = True
                    error = type(exc).__name__ + ': ' + str(exc)
                except Exception as exc:
                    error = type(exc).__name__ + ': ' + str(exc)
                attempt = dict(number=number, endpoint=url, seconds=time.perf_counter()-started,
                               http_status=None if response is None else response.status_code,
                               timed_out=timed_out, error=error)
                attempts.append(attempt)
                record_attempt(attempt, response)
                if timed_out:
                    with self.condition:
                        self.timeouts[url] += 1
                    print(f'[v7-timeout] endpoint={url} attempt={number}/4; ' +
                          ('retry on another available endpoint' if number < 4 else 'timeout budget exhausted'), flush=True)
                else:
                    return response, attempts, error
            previous = url
        return None, attempts, 'TIMEOUT_EXHAUSTED: initial request and three retries all timed out'
