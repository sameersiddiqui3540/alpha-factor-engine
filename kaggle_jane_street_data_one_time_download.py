import kagglehub

kagglehub.login()
download_path = r".\data\raw\kaggle-jane-street-data"

# Download latest version
path = kagglehub.competition_download(
    'jane-street-real-time-market-data-forecasting', 
    output_dir=download_path)

print("Path to competition files:", path)