"""
RESTful API for Agentic RAG System
==================================

This module implements a FastAPI RESTful API for the Agentic RAG System, providing endpoints
for data ingestion and query operations.

Author: Open Source Community
License: MIT
"""

from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import uvicorn
import os

# Import the core RAG system
from main import AgenticRAGSystem

app = FastAPI(title="Agentic RAG System API", version="1.0")

# Global RAG system instance
rag_system = None


class QueryModel(BaseModel):
    query: str


@app.on_event("startup")
async def startup_event():
    global rag_system
    
    # Configuration
    config = {
        "data_path": "./data",
        "storage_path": "./storage", 
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
    
    # Initialize and setup the RAG system
    rag_system = AgenticRAGSystem(config)
    rag_system.setup_system()
    

@app.post("/ingest", response_class=JSONResponse)
async def ingest_data(
    file: Optional[UploadFile] = File(None), 
    text: Optional[str] = Form(None)
):
    """
    Ingest text or PDF data into the RAG system.
    """
    try:
        if file:
            if file.content_type == 'application/pdf':
                # PDF file ingestion
                file_path = f"{rag_system.data_path}/{file.filename}"
                with open(file_path, "wb") as buffer:
                    buffer.write(file.file.read())
                rag_system.ingest_pdf(file_path)
                return {"message": f"Successfully ingested PDF: {file.filename}"}
            else:
                raise HTTPException(status_code=400, detail="Unsupported file type")
        elif text:
            # Text ingestion
            rag_system.ingest_text(text)
            return {"message": "Successfully ingested text."}
        else:
            raise HTTPException(status_code=400, detail="Text or PDF file must be provided.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_class=JSONResponse)
async def query_rag_system(query_data: QueryModel):
    """
    Accept user queries and return RAG-generated responses.
    """
    try:
        result = rag_system.query(query_data.query)
        return {
            "query": result.get("query"),
            "response": result.get("response"),
            "sources": result.get("sources"),
            "analysis": result.get("analysis")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_class=JSONResponse)
async def health_check():
    """
    Check system status and health.
    """
    try:
        stats = rag_system.get_system_stats()
        if stats["system_status"] == "healthy":
            return {
                "status": "healthy",
                "details": stats
            }
        else:
            return {
                "status": "not_initialized",
                "details": stats
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Running the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)

