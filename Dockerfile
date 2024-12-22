# Use Python 3.11-slim as base image
FROM python:3.12-slim

# Set working directory in container
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create a virtual environment in the container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --no-dev --no-interaction --no-ansi

# Copy your application code and model
COPY predictor.py model_v1.bin ./

# Expose the port your application runs on
EXPOSE 9696

# Updated CMD to use predictor.py instead of predict.py
CMD ["poetry", "run", "gunicorn", "--workers=4", "--bind=0.0.0.0:9696", "--timeout=60", "--access-logfile=-", "predictor:app"]