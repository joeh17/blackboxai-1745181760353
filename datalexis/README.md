# Datalexis

Datalexis is a scalable and interactive data analysis application. It supports loading large datasets, interactive visualization, basic machine learning models, data editing, and report generation. The app can be used as a desktop GUI, CLI tool, or web application.

## Features

- Load datasets with pandas, chunked reading, or Dask for scalability.
- Interactive GUI with tabs for overview, visualization, machine learning, data editing, and report generation.
- Web app interface built with Flask for easy access without GUI dependencies.
- Splash screen and intro visuals for enhanced user experience.
- Export analysis reports as HTML files.
- Dockerized for easy deployment.

## Installation

1. Clone the repository:
```
git clone <your-repo-url>
cd datalexis
```

2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

### CLI
Run the analysis script with options:
```
python datalexis/analysis.py --dataset path/to/data.csv --interactive
```

### Web App
Run the Flask web app:
```
python datalexis/web_app.py
```
Open your browser at http://localhost:8000

### Docker
Build and run the Docker container:
```
docker build -t datalexis-app datalexis/
docker run -p 8000:8000 datalexis-app
```

## Deployment

You can deploy the Docker container to cloud platforms like Heroku, AWS, or DigitalOcean.

## Contributing

Contributions are welcome! Please open issues or pull requests.

## License

MIT License
