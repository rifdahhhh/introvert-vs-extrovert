FROM tensorflow/serving:2.8.0

COPY ./outputs/serving_model /models/heart-failure-model
COPY ./config /model_config

ENV MODEL_NAME=heart-failure-model
ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV MODEL_BASE_PATH=/models
ENV PORT=${PORT:-8501}  

ENTRYPOINT ["/usr/bin/tf_serving_entrypoint.sh"]

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

EXPOSE ${PORT}
