FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install logbatcher
CMD ["python", "benchmark.py"]