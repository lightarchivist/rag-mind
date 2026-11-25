"""Minimal stand-in for FastAPI so the web app's endpoint functions can be
called directly in tests. Routing, validation and HTTP handling are NOT
exercised here; run the real app for that (see README smoke test)."""


class HTTPException(Exception):
    def __init__(self, status_code, detail=None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


class BackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, fn, *args, **kwargs):
        self.tasks.append((fn, args, kwargs))

    def run(self):
        for fn, args, kwargs in self.tasks:
            fn(*args, **kwargs)
        self.tasks.clear()


class UploadFile:
    def __init__(self, filename, file):
        self.filename = filename
        self.file = file


def File(default=None, **_):
    return default


class FastAPI:
    def __init__(self, **_):
        self.routes = []

    def _route(self, method, path, **_):
        def deco(fn):
            self.routes.append((method, path, fn.__name__))
            return fn
        return deco

    def get(self, path, **kw):
        return self._route("GET", path, **kw)

    def post(self, path, **kw):
        return self._route("POST", path, **kw)

    def delete(self, path, **kw):
        return self._route("DELETE", path, **kw)

    def mount(self, *a, **kw):
        pass
