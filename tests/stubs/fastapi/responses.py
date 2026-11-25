class HTMLResponse:
    pass


class StreamingResponse:
    def __init__(self, content, media_type=None, status_code=200):
        self.body_iterator = content
        self.media_type = media_type
        self.status_code = status_code

    def text(self):
        return "".join(self.body_iterator)
