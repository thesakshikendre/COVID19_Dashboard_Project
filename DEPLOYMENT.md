# COVID-19 Dashboard Deployment Guide

## Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning repository)

### Installation Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/COVID19_Dashboard_Project.git
   cd COVID19_Dashboard_Project
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv covid_env

   # Windows
   covid_env\Scripts\activate

   # macOS/Linux  
   source covid_env/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Data Pipeline**
   ```bash
   cd src
   python main_pipeline.py
   ```

5. **Launch Dashboard**
   ```bash
   cd ../dashboards
   streamlit run streamlit_app.py
   ```

The dashboard will be available at `http://localhost:8501`

## Production Deployment

### Docker Deployment

1. **Create Dockerfile**
   ```dockerfile
   FROM python:3.9-slim

   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt

   COPY . .
   EXPOSE 8501

   CMD ["streamlit", "run", "dashboards/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build and Run**
   ```bash
   docker build -t covid19-dashboard .
   docker run -p 8501:8501 covid19-dashboard
   ```

### Cloud Deployment Options

#### Streamlit Cloud
1. Push repository to GitHub
2. Connect to Streamlit Cloud
3. Deploy directly from repository

#### AWS/Azure/GCP
- Use container services (ECS, Container Instances, Cloud Run)
- Set up load balancing and auto-scaling
- Configure secure environment variables

## Data Pipeline Automation

### Scheduled Data Updates
```python
# cron job for daily updates
0 6 * * * cd /path/to/project && python src/main_pipeline.py
```

### GitHub Actions CI/CD
```yaml
name: Deploy Dashboard
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: python -m pytest tests/
      - name: Deploy
        run: # deployment commands
```

## Monitoring and Maintenance

### Performance Monitoring
- Set up application performance monitoring
- Monitor data pipeline execution times
- Track user engagement metrics

### Data Quality Checks
- Automated data validation
- Anomaly detection alerts
- Data freshness monitoring

### Security Considerations
- Implement authentication if needed
- Use HTTPS in production
- Secure sensitive configuration data
- Regular security updates

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Check Python path configuration
   - Verify all dependencies installed

2. **Data Loading Issues**
   - Check internet connectivity
   - Verify data source URLs
   - Review data processing logs

3. **Dashboard Performance**
   - Enable Streamlit caching
   - Optimize data loading
   - Consider data sampling for large datasets

4. **Memory Issues**
   - Monitor memory usage during pipeline execution
   - Implement data chunking for large files
   - Use appropriate data types

### Support
For technical support:
- Check project documentation
- Review GitHub issues
- Contact: your.email@example.com

## Configuration

### Environment Variables
```bash
# .env file
DATA_REFRESH_INTERVAL=24  # hours
CACHE_TIMEOUT=3600       # seconds
DEBUG_MODE=False
LOG_LEVEL=INFO
```

### Custom Settings
Modify `src/utils.py` CONFIG class for:
- Analysis parameters
- File paths
- API endpoints
- Visualization settings
