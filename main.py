from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import os
import sys

from app.pipeline import RAGPipeline
from app.schemas import IngestRequest, QueryRequest, QueryResponse
from app.file_processor import extract_text_from_file

# Debug Information: Server start aagumbodhu endha environment use pannudhu-nu kaatum
print(f"DEBUG: Current Python Executable: {sys.executable}")

app = FastAPI(title="Self-Improving RAG Pipeline", version="1.0.0")

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mount
static_path = "app/static"
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# Pipeline singleton initialization
pipeline: Optional[RAGPipeline] = None

@app.on_event("startup")
async def startup_event():
    global pipeline
    try:
        pipeline = RAGPipeline()
        print("INFO: RAG Pipeline initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to initialize RAG Pipeline: {e}")

@app.get("/")
async def serve_frontend():
    index_file = "app/static/index.html"
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Welcome to RAG API. Frontend not found in app/static/"}

@app.get("/health")
def health_check():
    return {"status": "ok", "pipeline_ready": pipeline is not None}

@app.post("/ingest")
def ingest_documents(request: IngestRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    doc_list = [doc.model_dump() for doc in request.documents]
    ids = pipeline.ingest_documents(doc_list)
    return {"ingested_ids": ids}

@app.post("/query", response_model=QueryResponse)
def query_documents(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    try:
        result = pipeline.run(query=request.query, top_k=request.top_k)
        return result
    except Exception as e:
        print(f"QUERY ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/ingest-files")
async def ingest_files(files: List[UploadFile] = File(...)):
    """Ingest multiple files (PDF, DOCX, TXT, JSON, etc.)"""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    documents = []
    errors = []
    
    for file in files:
        try:
            content = await file.read()
            # Extract text
            text = extract_text_from_file(file.filename, content)
            
            # Check if extraction returned an error message from file_processor
            if text.startswith("ERROR:"):
                print(f"PROCESSING ERROR for {file.filename}: {text}")
                errors.append(f"{file.filename}: {text}")
                continue

            if not text or not text.strip():
                errors.append(f"{file.filename}: Extracted text is empty")
                continue
            
            file_id = f"{file.filename.replace(' ', '_')}_{hash(content) % 10000}"
            
            documents.append({
                "id": file_id,
                "text": text,
                "metadata": {
                    "source": file.filename,
                    "content_type": file.content_type,
                }
            })
        except Exception as e:
            print(f"INTERNAL ERROR processing {file.filename}: {str(e)}")
            errors.append(f"{file.filename}: {str(e)}")
            continue

    if not documents:
        # 400 error varumbodhu exact reason-ah detail-ah kudukirom
        raise HTTPException(
            status_code=400, 
            detail={"message": "Could not extract valid text", "errors": errors}
        )
    
    ingested_ids = pipeline.ingest_documents(documents)
    
    return {
        "status": "success",
        "ingested_files_count": len(documents),
        "ingested_ids": ingested_ids
    }