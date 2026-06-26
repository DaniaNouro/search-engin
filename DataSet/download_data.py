import os
import sys
import locale

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["IR_DATASETS_HOME"] = r"D:\ir_datasets"

sys.stdout.reconfigure(encoding="utf-8")
locale.setlocale(locale.LC_ALL, "C")

import ir_datasets
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

# -----------------------------
# Config
# -----------------------------
BATCH_SIZE = 5000

# -----------------------------
# Main function
# -----------------------------
def download_documents():

    print("Loading Quora dataset...")

    dataset = ir_datasets.load("beir/quora/test")

    total_docs = dataset.docs_count()

    print(f"Total documents: {total_docs}")

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ir_search",
        user="postgres",
        password="password"
    )

    cursor = conn.cursor()

    batch = []
    inserted = 0

    with tqdm(total=total_docs, desc="Downloading Documents") as pbar:

        for doc in dataset.docs_iter():

            batch.append(
                (
                    str(doc.doc_id),
                    doc.text
                )
            )

            if len(batch) >= BATCH_SIZE:

                execute_values(
                    cursor,
                    """
                    INSERT INTO documents (doc_id, raw_text)
                    VALUES %s
                    ON CONFLICT (doc_id) DO NOTHING
                    """,
                    batch
                )

                conn.commit()

                inserted += len(batch)
                pbar.update(len(batch))

                batch.clear()

        # آخر دفعة
        if batch:

            execute_values(
                cursor,
                """
                INSERT INTO documents (doc_id, raw_text)
                VALUES %s
                ON CONFLICT (doc_id) DO NOTHING
                """,
                batch
            )

            conn.commit()

            inserted += len(batch)
            pbar.update(len(batch))

    cursor.close()
    conn.close()

    print(f"\nInserted {inserted} documents successfully.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    download_documents()