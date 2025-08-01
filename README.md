# Submission 2: Introvert vs. Extrovert Prediction

**Nama**: Rifdah Hansya Rofifah  
**Username Dicoding**: rifdahhr

| | Deskripsi |
| ----------- | ----------- |
| **Dataset** | [Extrovert vs. Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/) |
| **Masalah** | Permasalahan yang diangkat adalah bagaimana mengklasifikasikan seseorang ke dalam kepribadian introvert atau extrovert berdasarkan perilakunya. Klasifikasi ini bertujuan untuk memberikan insight atau personalisasi dalam berbagai bidang seperti pendidikan, pekerjaan, maupun sosial. |
| **Solusi Machine Learning** | Dengan memanfaatkan machine learning, sistem prediksi ini dapat mengklasifikasikan seseorang sebagai introvert atau extrovert secara otomatis berdasarkan data perilaku. Ini dapat membantu dalam pengambilan keputusan di bidang personalisasi layanan atau psikologi digital. |
| **Metode Pengolahan** | Data yang digunakan terdiri dari data kategorikal dan numerik. Data kategorikal ditransformasikan menggunakan one-hot encoding, sedangkan data numerik dinormalisasi ke dalam rentang yang seragam. |
| **Arsitektur Model** | Model dibangun menggunakan arsitektur sederhana dengan beberapa Dense layer dan Dropout layer, serta satu output layer dengan aktivasi sigmoid untuk klasifikasi biner. |
| **Metrik Evaluasi** | Evaluasi model dilakukan menggunakan AUC, Precision, Recall, BinaryAccuracy, TruePositive, FalsePositive, TrueNegative, dan FalseNegative. |
| **Performa Model** | Model menghasilkan akurasi biner sebesar 87% pada data pelatihan dan 85% pada data validasi. Ini menunjukkan bahwa model cukup baik dalam melakukan klasifikasi kepribadian. |
| **Opsi Deployment** | Proyek ini dideploy menggunakan platform Railway yang mendukung deployment layanan machine learning berbasis API. |
| **Web App** | <https://mlops-submission-production.up.railway.app/v1/models/personality_model> |
| **Monitoring** | Monitoring dilakukan menggunakan Prometheus dan Grafana. Lima metrik utama yang dipantau meliputi:<br>1. `:tensorflow:serving:request_count` - jumlah request prediksi<br>2. `:tensorflow:serving:request_latency_sum` dan `request_latency_count` - untuk rata-rata latensi<br>3. `:tensorflow:cc:saved_model:load_latency` - waktu loading model<br>4. `:tensorflow:core:graph_run_time_usecs` - waktu eksekusi graf TensorFlow<br>5. `up` - status server model. Semua metrik divisualisasikan di Grafana untuk memastikan performa sistem terjaga. |
