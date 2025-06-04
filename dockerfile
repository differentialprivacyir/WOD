# ---- Base image ------------------------------------------------------------
FROM python:3.6.9-buster

# ---- Install Python dependencies ------------------------------------------
RUN pip install --no-cache-dir pandas scikit-learn xxhash progressbar tabulate

# ---- Create working directory ---------------------------------------------
WORKDIR /app

# ---- Copy source code ------------------------------------------------------
COPY ./ ./

# ---- Default command -------------------------------------------------------
CMD ["python", "main.py"]
