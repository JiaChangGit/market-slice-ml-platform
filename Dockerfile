FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 graphviz locales \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip wheel "setuptools<82" \
    && pip install --no-cache-dir . \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV DATA_ROOT=/app/data
ENV NO_NETWORK=1
VOLUME ["/app/data"]
ENTRYPOINT ["market-ml"]
