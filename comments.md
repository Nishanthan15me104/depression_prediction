1. LOOK IN UI for fast api end point 
http://localhost:8000/docs#


default key : dev-secret-key-1234 
env key : my-super-secret-production-key : use ehn runniin gin local

2. Access the Dashboard FOR MLFLOW: 
```
[INFO] Listening at: http://127.0.0.1:5000

    In the "Connection" URL box, type exactly: http://prometheus:9090



3. API METRICS BY Prometheus SEE TI HERE :

http://localhost:8000/metrics

4. Look at Grafana


http://localhost:3000
````


TO START FAST API
python run_api.py



click authorize and give key : 
and save it 



NEED TO BE IMPLEMENTED PALN FROM HERE :::::::::::::::::::

The Observability Roadmap1.Establish Baselines:Current Step.Run your API locally and hit the /predict endpoint with 50-100 test requests. Look at your /metrics page to find your Average Latency. If your average is 200ms, you now know that an alert should trigger if it hits 500ms.2.Connect Prometheus + Grafana:The Visual Layer.Launch Prometheus and Grafana via Docker Compose. Connect Grafana to your API’s /metrics stream. Create a dashboard with three panels: Request Volume, Error Rate (4xx/5xx), and p95 Latency.3.Integrate Evidently AI:The Model Quality Layer.Add a "Reference Data" CSV (from your training set) to your repo. Update your API to collect the last 50 production inputs in a small buffer. Use Evidently to compare this buffer against the reference to detect Data Drift.4.Deploy to Azure (ACA):The Infrastructure Layer.Push your updated image to GHCR. Deploy to Azure Container Apps. Enable Application Insights during setup; this automatically starts capturing container logs and CPU/RAM usage without extra code.5.Set Azure Alerts:The Safety Net.In the Azure Portal, set a Budget Alert (to prevent surprise costs) and a Metric Alert. If the container restarts or the memory usage exceeds 80%, Azure will email you immediately.