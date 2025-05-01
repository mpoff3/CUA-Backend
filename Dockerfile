# syntax=docker/dockerfile:1
############################################################
# 1) Base image with browsers & system deps pre-installed  #
############################################################
FROM mcr.microsoft.com/playwright/python:v1.50.0-jammy

#########################
# 2) Basic environment  #
#########################
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

####################
# 3) Application   #
####################
WORKDIR /app

# Only copy requirements first so Docker layer caching works
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the source code
COPY . .

# Ensure non-root user (pwuser) owns application directory so Chromium sandbox works
RUN chown -R pwuser:pwuser /app

USER pwuser

##############################
# 4) Expose & default cmd    #
##############################
# Render injects $PORT (e.g. 10000).  Fallback to 8000 for local dev
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]