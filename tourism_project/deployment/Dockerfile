# Base image
FROM python:3.9

# System setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Create non-root user and app dir
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Install dependencies
COPY --chown=user requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY --chown=user app.py ./

# Expose and run
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
