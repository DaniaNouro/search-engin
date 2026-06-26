"""
Module Name: store_clean.py

Purpose:
Reads raw documents from PostgreSQL,
applies light preprocessing,
and stores results into processed_documents table.

Optimized for large datasets (500K+ documents).
"""

import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

from services.preprocessing.base_cleaner import light_cleaning_pipeline

BATCH_SIZE = 10000


def run_light_preprocessing_pipeline():

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ir_search",
        user="postgres",
        password="password"
    )

    # Cursor للقراءة
    read_cursor = conn.cursor()

    # Cursor للكتابة
    write_cursor = conn.cursor()

    # عدد الوثائق الكلي
    read_cursor.execute(
        "SELECT COUNT(*) FROM documents;"
    )

    total_docs = read_cursor.fetchone()[0]

    print(f"\nFound {total_docs:,} documents")
    print("Starting preprocessing pipeline...\n")

    # قراءة الوثائق
    read_cursor.execute(
        """
        SELECT doc_id, raw_text
        FROM documents
        """
    )

    with tqdm(
        total=total_docs,
        desc="Preprocessing",
        unit="docs"
    ) as pbar:

        while True:

            rows = read_cursor.fetchmany(BATCH_SIZE)

            if not rows:
                break

            batch = []

            for doc_id, raw_text in rows:

                if raw_text is None:
                    continue

                clean_text = light_cleaning_pipeline(raw_text)

                batch.append(
                    (
                        doc_id,
                        clean_text
                    )
                )

            if batch:

                execute_values(
                    write_cursor,
                    """
                    INSERT INTO processed_documents
                    (doc_id, clean_text)
                    VALUES %s
                    ON CONFLICT (doc_id)
                    DO NOTHING
                    """,
                    batch,
                    page_size=BATCH_SIZE
                )

                conn.commit()

            pbar.update(len(rows))

    read_cursor.close()
    write_cursor.close()
    conn.close()

    print("\nPreprocessing completed successfully.")
    print("All cleaned documents stored in processed_documents.")


if __name__ == "__main__":
    run_light_preprocessing_pipeline()