FROM python:3.11-slim

# opencv needs libGL and libglib even in headless use.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The server and package code live under src/. The sample data under data/ is
# not copied: it is ~400 MB and only local_evaluator.py needs it.
COPY src/ ./

EXPOSE 9053

CMD ["python", "api.py"]
