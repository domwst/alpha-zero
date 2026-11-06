FROM alpine:3.23

RUN apk add --no-cache python3 py3-pip
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
USER 1000:1000
ENTRYPOINT ["/opt/venv/bin/python3"]
