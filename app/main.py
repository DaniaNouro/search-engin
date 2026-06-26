"""
Module Name: main.py
Purpose: The central entrypoint for the FastAPI backend application.
         Registers sub-routers, configures CORS middleware, and boots the Uvicorn server.
"""

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# استيراد الموجهات من المجلد الفرعي
from app.endpoints import tfidf_router, bm25_router, bert_router, evaluation_router,hybrid_router

app = FastAPI(
    title="Advanced IR Search Engine Core APIs",
    description="Service-Oriented Architecture (SOA) backend powering lexical and semantic search engines.",
    version="2.0.0"
)

# 🌐 إعدادات الـ CORS لحل مشكلة الحظر الأمني عند اتصال واجهة Streamlit بالسيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # في بيئة التطوير نفتحها لجميع الواجهات
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تسجيل المسارات داخل التطبيق المركزي
app.include_router(tfidf_router.router)
app.include_router(bm25_router.router)
app.include_router(bert_router.router)
app.include_router(evaluation_router.router)
app.include_router(hybrid_router.router)


@app.get("/", tags=["Root Component"])
def read_root():
    """Health check endpoint to ensure API gateway is alive."""
    return {
        "status": "Healthy",
        "message": "Welcome to the IR Search Engine Core API Gateway. All services operational."
    }


if __name__ == "__main__":
    # تشغيل السيرفر محلياً على البورت 8000
    uvicorn.run("main:app", host="127.0.0.1", port=8085, reload=True)