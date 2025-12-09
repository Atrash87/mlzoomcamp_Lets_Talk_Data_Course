FROM agrigorev/model-2025-hairstyle:v1

WORKDIR /app
COPY lambda_function.py .

RUN pip install --no-cache-dir numpy pillow onnxruntime

CMD ["/bin/bash"]
