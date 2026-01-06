import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

if MAPBOX_TOKEN is None:
    raise ValueError("❌ MAPBOX_TOKEN not found in .env")

# Image parameters
IMAGE_SIZE = 224
ZOOM_LEVEL = 18
MAP_STYLE = "mapbox/satellite-v9"

# Paths
TRAIN_CSV = "data/processed/train_sampled.csv"
TRAIN_IMG_DIR = "data/images/train"

os.makedirs(TRAIN_IMG_DIR, exist_ok=True)

def fetch_image(lat, lon, save_path, retries=3):
    url = (
        f"https://api.mapbox.com/styles/v1/{MAP_STYLE}/static/"
        f"{lon},{lat},{ZOOM_LEVEL}/{IMAGE_SIZE}x{IMAGE_SIZE}"
        f"?access_token={MAPBOX_TOKEN}"
    )

    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                return True

            time.sleep(1)

        except Exception:
            time.sleep(1)

    return False

train_df = pd.read_csv(TRAIN_CSV)

print(f"🔹 Training samples to download: {len(train_df)}")

failed_ids = []

for _, row in tqdm(train_df.iterrows(), total=len(train_df)):
    img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id']}.png")

    # Resume-safe
    if os.path.exists(img_path):
        continue

    success = fetch_image(
        lat=row["lat"],
        lon=row["long"],
        save_path=img_path
    )

    if not success:
        failed_ids.append(row["id"])

    time.sleep(0.1)  # rate limiting

if failed_ids:
    pd.DataFrame({"id": failed_ids}).to_csv(
        "data/images/failed_train_images.csv",
        index=False
    )
    print(f"⚠️ Failed downloads: {len(failed_ids)}")
else:
    print("✅ All images downloaded successfully")
