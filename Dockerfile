FROM tensorflow/serving:2.8.0

# Salin seluruh folder model ke dalam image Docker
# Pastikan struktur lokalmu adalah: ./outputs/serving_model/1/saved_model.pb
COPY ./outputs/serving_model /models/personality_model

# Salin konfigurasi monitoring (jika digunakan)
COPY ./config /model_config

# Konfigurasi environment
ENV MODEL_NAME=personality_model
ENV MODEL_BASE_PATH=/models
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8080

# Buat entrypoint script agar menjalankan tensorflow_model_server dengan opsi yang benar
RUN echo '#!/bin/bash \n\
echo "Starting TensorFlow Serving on port ${PORT}" \n\
tensorflow_model_server \\\n\
  --port=8500 \\\n\
  --rest_api_port=${PORT} \\\n\
  --model_name=${MODEL_NAME} \\\n\
  --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \\\n\
  --monitoring_config_file=${MONITORING_CONFIG} \\\n\
  "$@"' > /usr/bin/tf_serving_entrypoint.sh \
  && chmod +x /usr/bin/tf_serving_entrypoint.sh

# Jalankan entrypoint
ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]

EXPOSE 8080