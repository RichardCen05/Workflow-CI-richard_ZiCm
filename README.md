# Workflow-CI-richard_ZiCm

Repository kriteria 3 untuk menjalankan MLflow Project menggunakan GitHub Actions.

Workflow sudah menghindari pemanggilan `secrets` langsung pada kondisi `if`; secrets didaftarkan sebagai environment variable dan statusnya dievaluasi melalui step `docker_secrets`.
