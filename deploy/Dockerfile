FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY deploy/requirements.txt /app/deploy/requirements.txt
RUN pip install -r /app/deploy/requirements.txt
RUN useradd -m -u 1000 appuser
COPY app.py /app/app.py
COPY app_components /app/app_components
COPY app_services /app/app_services
COPY assets /app/assets
COPY .streamlit /app/.streamlit
COPY src /app/src
COPY config/config.py /app/config/config.py
COPY config/app_release.json /app/config/app_release.json
RUN mkdir /app/models && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["sh","-c","python -m app_services.lab && exec python -m streamlit run app.py --server.address 0.0.0.0 --server.port 7860"]
