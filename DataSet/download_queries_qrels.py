import ir_datasets
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

BATCH_SIZE = 5000


def download_queries_qrels():
    print("Loading Quora dataset...")

    dataset = ir_datasets.load("beir/quora/test")

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="ir_search",
        user="postgres",
        password="password"
    )
    cursor = conn.cursor()

    # =========================
    # QUERIES
    # =========================
    print("Loading queries...")

    queries = []
    for q in tqdm(dataset.queries_iter(), total=dataset.queries_count()):
        queries.append((q.query_id, q.text))

        if len(queries) >= BATCH_SIZE:
            execute_values(
                cursor,
                """
                INSERT INTO queries (query_id, query_text)
                VALUES %s
                ON CONFLICT DO NOTHING
                """,
                queries
            )
            conn.commit()
            queries = []

    if queries:
        execute_values(
            cursor,
            """
            INSERT INTO queries (query_id, query_text)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            queries
        )
        conn.commit()

    # =========================
    # QRELS
    # =========================
    print("Loading qrels...")

    qrels = []
    qrels_iter = dataset.qrels_dict()

    # BEIR structure: query_id -> doc_id -> relevance
    for query_id, docs in tqdm(qrels_iter.items()):
        for doc_id, rel in docs.items():
            qrels.append((query_id, doc_id, rel))

            if len(qrels) >= BATCH_SIZE:
                execute_values(
                    cursor,
                    """
                    INSERT INTO qrels (query_id, doc_id, relevance)
                    VALUES %s
                    ON CONFLICT DO NOTHING
                    """,
                    qrels
                )
                conn.commit()
                qrels = []

    if qrels:
        execute_values(
            cursor,
            """
            INSERT INTO qrels (query_id, doc_id, relevance)
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            qrels
        )
        conn.commit()

    cursor.close()
    conn.close()

    print("Done 🚀")


if __name__ == "__main__":
    download_queries_qrels()