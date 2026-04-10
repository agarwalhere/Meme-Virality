import os
import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    # Fallback if tqdm is not installed — keeps functionality without progress bars
    def tqdm(iterable, total=None, desc=None):
        return iterable
try:
    import easyocr
except Exception:
    easyocr = None
try:
    import pytesseract
    from PIL import Image
except Exception:
    pytesseract = None
    Image = None
import argparse


def perform_ocr(image_folder, csv_file):
    print("🔄 Initializing OCR engine...")
    reader = None
    engine = None

    if easyocr is not None:
        try:
            reader = easyocr.Reader(['en'], gpu=True)
            engine = 'easyocr'
            print("ℹ️ Using easyocr for text extraction (GPU enabled).")
        except Exception as e:
            print(f"⚠️ easyocr initialization failed: {e}")
            if pytesseract is not None:
                engine = 'pytesseract'
                print("ℹ️ Falling back to pytesseract.")
            else:
                raise RuntimeError(
                    "easyocr failed to initialize (network/SSL). Install pytesseract and the Tesseract binary or fix your SSL certificates."
                )
    elif pytesseract is not None:
        engine = 'pytesseract'
        print("ℹ️ easyocr not available — using pytesseract.")
    else:
        raise RuntimeError(
            "No OCR engine available: install `easyocr` (recommended) or `pytesseract` and the Tesseract binary."
        )

    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")

    df = pd.read_csv(csv_file, encoding="ISO-8859-1")

    if "File Name" not in df.columns:
        raise ValueError("CSV must contain a column named 'File Name'")

    df["Extracted Text"] = ""

    rows_to_delete = []

    print("🔍 Performing OCR on images...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="OCR Progress"):
        image_name = str(row["File Name"])
        image_path = os.path.join(image_folder, image_name)

        if os.path.isfile(image_path):
                try:
                    if engine == 'easyocr':
                        results = reader.readtext(image_path)
                        extracted_text = " ".join([result[1] for result in results]) if results else ""
                    elif engine == 'pytesseract':
                        img = Image.open(image_path)
                        text = pytesseract.image_to_string(img)
                        extracted_text = text.strip() if text else ""
                    else:
                        extracted_text = ""

                    if extracted_text:
                        df.at[index, "Extracted Text"] = extracted_text
                    else:
                        rows_to_delete.append(index)
                        try:
                            os.remove(image_path)
                            print(f"❌ No text found. Deleted: {image_name}")
                        except Exception:
                            print(f"❌ No text found for {image_name} (could not delete file)")

                except Exception as e:
                    rows_to_delete.append(index)
                    try:
                        os.remove(image_path)
                    except Exception:
                        pass
                    print(f"❌ Error processing {image_name}: {e}")
        else:
            rows_to_delete.append(index)
            print(f"❌ File not found: {image_name}")

    # Remove invalid rows
    df.drop(rows_to_delete, inplace=True)
    df.reset_index(drop=True, inplace=True)

    print("📝 Renaming images to maintain numbering...")
    # Pass 1: rename everything to temporary names to prevent collisions
    temp_names = {}
    for new_index, row in tqdm(df.iterrows(), total=len(df), desc="Renaming Pass 1"):
        old_name = str(row["File Name"])
        old_path = os.path.join(image_folder, old_name)

        if os.path.isfile(old_path):
            new_name = f"{new_index + 1}_{old_name.split('_', 1)[-1]}"
            if old_name == new_name:
                temp_names[new_index] = old_name
            else:
                temp_idx = f".temp_{new_index}_{old_name}"
                temp_path = os.path.join(image_folder, temp_idx)
                os.rename(old_path, temp_path)
                temp_names[new_index] = temp_idx

    # Pass 2: rename temporary names to final names
    for new_index, row in tqdm(df.iterrows(), total=len(df), desc="Renaming Pass 2"):
        if new_index in temp_names:
            temp_name = temp_names[new_index]
            old_name = str(row["File Name"])
            new_name = f"{new_index + 1}_{old_name.split('_', 1)[-1]}"
            
            if temp_name != new_name:
                temp_path = os.path.join(image_folder, temp_name)
                new_path = os.path.join(image_folder, new_name)
                
                if os.path.exists(new_path):
                    try: os.replace(temp_path, new_path)
                    except: pass
                else:
                    os.rename(temp_path, new_path)
                
            df.at[new_index, "File Name"] = new_name

    df.to_csv(csv_file, index=False)
    print(f"✅ Updated CSV saved as: {csv_file}")

    return df


def main():
    parser = argparse.ArgumentParser(description="Run OCR on images listed in CSV.")

    # Make default paths relative to this script file so running from any CWD works
    base_dir = os.path.dirname(os.path.abspath(__file__))
    default_images = os.path.join(base_dir, "custom_reddit_dataset", "memes")
    default_csv = os.path.join(base_dir, "custom_reddit_dataset", "data.csv")

    parser.add_argument("--images", type=str, default=default_images,
                        help="Path to image folder")
    parser.add_argument("--csv", type=str, default=default_csv,
                        help="Path to metadata CSV file")

    args = parser.parse_args()

    perform_ocr(args.images, args.csv)


if __name__ == "__main__":
    main()