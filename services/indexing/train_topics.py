"""
Module Name: train_topics.py
Purpose: Offline Topic Detection Pipeline using LDA.
         Fetches processed documents from PostgreSQL, trains LDA model,
         assigns topic_id to each document using high-performance Bulk Update,
         and saves the model and vectorizer to disk.
"""

import os
import sys
import joblib
import psycopg2
from psycopg2.extras import execute_values
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

STORE_DIR = os.path.join("data", "vector_store")
TOPIC_MODEL_PATH = os.path.join(STORE_DIR, "topic_model.joblib")
TOPIC_VECTORIZER_PATH = os.path.join(STORE_DIR, "topic_vectorizer.joblib")

N_TOPICS = 10
SAMPLE_SIZE = 50000  # حجم العينة العشوائية الممثلة للتدريب الأولي


def run_topic_detection_pipeline():
    print("=== Starting Topic Detection Offline Pipeline ===")
    os.makedirs(STORE_DIR, exist_ok=True)

    # 1. الاتصال بقاعدة البيانات
    try:
        print("⏳ Connecting to PostgreSQL...")
        conn = psycopg2.connect(
            host="localhost", port=5432, database="ir_search",
            user="postgres", password="password"  # ⚠️ تأكدي من مطابقتها لبياناتكِ
        )
        cursor = conn.cursor()

        # 2. جلب عينة عشوائية مكثفة لتدريب الموديل
        print(f"📥 Fetching {SAMPLE_SIZE} documents for training...")
        cursor.execute("""
            SELECT doc_id, clean_text 
            FROM processed_documents 
            WHERE clean_text IS NOT NULL AND clean_text != ''
            ORDER BY RANDOM()
            LIMIT %s;
        """, (SAMPLE_SIZE,))
        train_rows = cursor.fetchall()

        if not train_rows:
            print("❌ No documents found in processed_documents table!")
            return

        train_ids = [row[0] for row in train_rows]
        train_texts = [row[1] for row in train_rows]
        print(f"✅ Successfully fetched {len(train_texts)} documents for training.")

        # 3. بناء الـ CountVectorizer (تثبيت الـ Vocabulary)
        print("🔤 Fitting CountVectorizer (Building Vocabulary)...")
        vectorizer = CountVectorizer(
            max_features=2000,   # حصر القاموس بأهم 2000 كلمة لمنع التشتت والبطء
            stop_words='english',
            min_df=5,            # تجاهل الكلمات النادرة جداً
            max_df=0.95          # تجاهل الكلمات الشائعة والمكررة بكثرة
        )
        X_train = vectorizer.fit_transform(train_texts)
        print(f"✅ Vocabulary size: {len(vectorizer.get_feature_names_out())}")

        # 4. تدريب خوارزمية الـ LDA (Latent Dirichlet Allocation)
        print(f"🧠 Training LDA Model with {N_TOPICS} topics...")
        lda = LatentDirichletAllocation(
            n_components=N_TOPICS,
            random_state=42,
            n_jobs=-1  # استغلال جميع أنوية الـ CPU لتسريع المعالجة الرياضية
        )
        lda.fit(X_train)
        print("✅ LDA training completed successfully!")

        # 5. استعراض المواضيع المكتشفة للتأكد من الترابط الدلالي
        print("\n📊 Discovered Topics Breakdown:")
        feature_names = vectorizer.get_feature_names_out()
        for topic_id, topic in enumerate(lda.components_):
            top_words = [feature_names[i] for i in topic.argsort()[-8:][::-1]]
            print(f"   Topic {topic_id}: {' | '.join(top_words)}")

        # 6. تمرير النموذج على كامل الـ 522K وثيقة وتحديث الـ DB بدفعات خاطفة
        print("\n⚡ Assigning topics to ALL documents in database (High-Performance Pipeline)...")

        # إنشاء العمود هيكلياً في الجدول إذا لم يكن موجوداً مسبقاً
        cursor.execute("""
            ALTER TABLE processed_documents 
            ADD COLUMN IF NOT EXISTS topic_id INTEGER;
        """)
        conn.commit()

        # جلب البيانات كاملة عبر كورسور مخصص للـ Streaming
        BATCH_SIZE = 10000
        cursor.execute("""
            SELECT doc_id, clean_text 
            FROM processed_documents 
            WHERE clean_text IS NOT NULL AND clean_text != '';
        """)

        processed = 0
        while True:
            rows = cursor.fetchmany(BATCH_SIZE)
            if not rows:
                break

            batch_ids = [row[0] for row in rows]
            batch_texts = [row[1] for row in rows]

            # تحويل النصوص الحالية بناءً على الـ Vocabulary المثبت واستنتاج التوزيع
            X_batch = vectorizer.transform(batch_texts)
            topic_distributions = lda.transform(X_batch)
            topic_ids = topic_distributions.argmax(axis=1)

            # 🚀 [التعديل الهندسي]: تحضير البيانات وتحديث الـ DB دفعة واحدة عبر سطر واحد من الـ RAM
            update_data = [(int(topic_id), doc_id) for doc_id, topic_id in zip(batch_ids, topic_ids)]

            update_query = """
                UPDATE processed_documents 
                SET topic_id = data.topic_id 
                FROM (VALUES %s) AS data(topic_id, doc_id) 
                WHERE processed_documents.doc_id = data.doc_id
            """

            with conn.cursor() as batch_cursor:
                execute_values(batch_cursor, update_query, update_data)
            conn.commit()

            processed += len(rows)
            print(f"   ├── Processed and updated {processed} documents...")

        # 7. حفظ مخرجات التدريب لاستخدامها الفوري في سرفر الـ FastAPI أوفلاين
        print(f"\n💾 Saving LDA model to: {TOPIC_MODEL_PATH}")
        joblib.dump(lda, TOPIC_MODEL_PATH)

        print(f"💾 Saving vectorizer to: {TOPIC_VECTORIZER_PATH}")
        joblib.dump(vectorizer, TOPIC_VECTORIZER_PATH)

        # 8. استخراج إحصائيات التوزيع الإجمالي لعرضها أمام اللجنة
        print("\n📊 Final Topic Distribution in PostgreSQL Database:")
        cursor.execute("""
            SELECT topic_id, COUNT(*) as count 
            FROM processed_documents 
            WHERE topic_id IS NOT NULL
            GROUP BY topic_id 
            ORDER BY topic_id;
        """)
        stats = cursor.fetchall()
        total = sum(row[1] for row in stats)
        for topic_id, count in stats:
            percentage = (count / total) * 100
            print(f"   Topic {topic_id}: {count:,} docs ({percentage:.1f}%)")

        cursor.close()
        conn.close()

        print("\n🎉 [SUCCESS] Topic Detection Offline Pipeline completed flawlessly!")

    except Exception as e:
        print(f"❌ Critical Pipeline Error: {str(e)}")
        raise


if __name__ == "__main__":
    run_topic_detection_pipeline()