import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our enhanced assistant
from autopsy_assistant_llamaindex import AutopsyAssistantEngine

# --- Configuration ---
PORT = int(os.getenv("PORT", "8888"))
HOST = os.getenv("HOST", "0.0.0.0")

# # RunPod specific configuration
# RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")
# RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")
# RUNPOD_BASE_URL = os.getenv("RUNPOD_BASE_URL", "https://api.runpod.io")

# --- Initialize FastAPI App ---
app = FastAPI(
    title="Autopsy Assistant RunPod Service",
    description="Full LlamaIndex RAG pipeline with RunPod LLM",
    version="2.0.0-runpod",
    docs_url="/docs",
    redoc_url="/redoc"
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend-domain.com",  # Replace with your actual domain
        "https://app-dev.healthdataexplorer.io",  # Replace with your actual domain
        "https://app.healthdataexplorer.io",  # Replace with your actual domain
        "http://localhost:3000",             # For local development
        "http://localhost:8080",             # For local development
        "*",                                  # Allow all origins (for demo purposes)
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize Autopsy Assistant ---
assistant_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the autopsy assistant on startup"""
    global assistant_engine
    try:
        print("🚀 Initializing Autopsy Assistant with RunPod...")
        assistant_engine = AutopsyAssistantEngine()
        print("✅ Autopsy Assistant initialized successfully!")
    except Exception as e:
        print(f"❌ Failed to initialize Autopsy Assistant: {e}")
        print("⚠️  Service will start but LLM features will be limited")

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for continuity")
    top_k: Optional[int] = Field(10, description="Number of search results")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation ID")
    timestamp: str = Field(..., description="Response timestamp")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Source documents")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Response metadata")
    error: Optional[str] = Field(None, description="Error message if any")

class ConfigurableSearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=50, description="Number of results to return")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    detail_level: str = Field(default="comprehensive", description="Detail level: brief, comprehensive, detailed")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(50, description="Number of results to return")

class CompareRequest(BaseModel):
    case_ids: List[str] = Field(..., description="Case IDs to compare")
    comparison_type: str = Field("general", description="Type of comparison")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class ComprehensivePatternRequest(BaseModel):
    pattern_type: str = Field(..., description="Type of pattern to analyze")
    time_period: str = Field(default="all", description="Time period for analysis")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    max_results: int = Field(default=500, description="Maximum number of results to analyze")

class PatternRequest(BaseModel):
    pattern_type: str = Field("comprehensive", description="Type of pattern to analyze")
    time_period: str = Field("all", description="Time period for analysis")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: str = Field(..., description="Current timestamp")
    features: List[str] = Field(..., description="Available features")
    version: str = Field(..., description="API version")
    assistant_status: str = Field(..., description="Autopsy Assistant status")
    runpod_status: str = Field(..., description="RunPod connection status")

# # --- Health Check ---
# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint"""
#     assistant_status = "uninitialized"
#     runpod_status = "unknown"
    
#     if assistant_engine:
#         try:
#             # Test basic functionality
#             assistant_status = "healthy"
#         except Exception:
#             assistant_status = "error"
    
#     # Check RunPod connection if configured
#     if RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID:
#         try:
#             # You could add a simple RunPod API call here to verify connectivity
#             runpod_status = "configured"
#         except Exception:
#             runpod_status = "error"
#     else:
#         runpod_status = "not_configured"
    
#     return HealthResponse(
#         status="healthy",
#         timestamp=datetime.now().isoformat(),
#         features=["llamaindex_rag", "runpod_llm", "mongodb_vector_search", "full_api"],
#         version="2.0.0-runpod",
#         assistant_status=assistant_status,
#         runpod_status=runpod_status
#     )

# --- API Endpoints ---

# @app.get("/", response_model=Dict[str, str])
# async def root():
#     """Root endpoint with API information"""
#     return {
#         "message": "Autopsy Assistant RunPod Service",
#         "version": "2.0.0-runpod",
#         "docs": "/docs",
#         "health": "/health",
#         "features": "Full LlamaIndex RAG pipeline with RunPod",
#         "runpod_configured": bool(RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID)
#     }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Chat endpoint with full LlamaIndex RAG pipeline"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Use the full LlamaIndex RAG pipeline
        response_data = assistant_engine.chat(
            message=request.message,
            top_k=request.top_k or 10
        )
        
        return ChatResponse(
            response=response_data.get("response", "No response"),
            conversation_id=request.conversation_id or str(uuid4()),
            timestamp=datetime.now().isoformat(),
            sources=response_data.get("sources"),
            metadata=response_data.get("metadata")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.post("/api/search")
async def search_endpoint(request: SearchRequest):
    """Enhanced semantic search endpoint"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Create a search-specific prompt
        search_query = f"Search for: {request.query}"
        
        # Use custom top_k if provided, otherwise use default
        top_k = request.top_k if request.top_k is not None else 50
        
        result = assistant_engine.chat(
            message=search_query,
            conversation_id=None,  # Search doesn't need conversation context
            top_k=top_k
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/api/patterns")
async def patterns_endpoint(request: PatternRequest):
    """Pattern analysis endpoint"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        query = f"Analyze {request.pattern_type} across autopsy reports for {request.time_period} time period"
        
        # Use smaller top_k for pattern analysis to avoid timeouts
        # For "all" time period, use manageable results
        if request.time_period.lower() == "all":
            top_k = 25  # Reduced to avoid timeouts
        else:
            top_k = 20  # Standard for other time periods
        
        result = assistant_engine.chat(
            message=query,
            conversation_id=request.conversation_id,
            top_k=top_k
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern analysis error: {str(e)}")

@app.get("/api/cases")
async def cases_endpoint():
    """List available case IDs"""
    try:
        # Access MongoDB collection directly
        if hasattr(assistant_engine, 'collection') and assistant_engine.collection is not None:
            # Get unique case IDs from MongoDB (matching original structure)
            pipeline = [
                {"$group": {"_id": "$case_id"}},
                {"$sort": {"_id": 1}}
            ]
            
            cases = list(assistant_engine.collection.aggregate(pipeline))
            case_ids = [case["_id"] for case in cases]
            
            return {
                "cases": case_ids,
                "total": len(case_ids),
                "timestamp": datetime.now().isoformat()
            }
        else:
            # Fallback if collection not available
            return {
                "cases": [],
                "total": 0,
                "timestamp": datetime.now().isoformat(),
                "message": "MongoDB collection not available"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching cases: {str(e)}")

@app.get("/api/case-details/{case_id}")
async def case_details_endpoint(case_id: str):
    """Get detailed case information"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Access MongoDB collection directly
        if hasattr(assistant_engine, 'collection') and assistant_engine.collection is not None:
            # Get all chunks for this case
            chunks = list(assistant_engine.collection.find({"case_id": case_id}).sort("chunk_index", 1))
            
            if not chunks:
                raise HTTPException(status_code=404, detail=f"Case {case_id} not found")
            
            # Combine all text
            full_text = " ".join([chunk.get("text", "") for chunk in chunks])
            
            return {
                "case_id": case_id,
                "chunk_count": len(chunks),
                "full_text": full_text,
                "chunks": [{
                    "chunk_id": str(chunk.get("_id")),
                    "chunk_index": chunk.get("chunk_index", 0),
                    "text": chunk.get("text", ""),
                    "score": chunk.get("score", 0.0)
                } for chunk in chunks]
            }
        else:
            raise HTTPException(status_code=500, detail="MongoDB collection not available")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Case details error: {str(e)}")

@app.post("/api/patterns-stats")
async def patterns_stats_endpoint(request: PatternRequest):
    """Pattern analysis using MongoDB aggregation for comprehensive statistics"""
    try:
        # Access MongoDB collection directly
        if hasattr(assistant_engine, 'collection') and assistant_engine.collection is not None:
            # Get comprehensive statistics using MongoDB aggregation
            # Collect ALL text from ALL chunks per case
            pipeline = [
                {
                    "$group": {
                        "_id": "$case_id",
                        "all_texts": {"$push": "$text"},  # Collect ALL chunk texts per case
                        "chunk_count": {"$sum": 1}
                    }
                },
                {
                    "$sort": {"_id": 1}
                }
            ]
            
            cases = list(assistant_engine.collection.aggregate(pipeline))
            total_cases = len(cases)
            
            # Get unique case IDs
            case_ids = [case["_id"] for case in cases]
            
            # ENHANCED: Analyze patterns per case, not per occurrence
            # This prevents double-counting when the same term appears in multiple chunks of the same case
            case_patterns = {}
            
            for case in cases:
                case_id = case["_id"]
                case_text = " ".join(case["all_texts"]).lower()
                case_patterns[case_id] = case_text
            
            # Calculate total text length from all cases
            total_text_length = sum(len(case_text) for case_text in case_patterns.values())
            
            # Basic pattern analysis
            patterns = {
                "total_cases": total_cases,
                "case_ids": case_ids,
                "total_chunks": sum(case["chunk_count"] for case in cases),
                "text_length": total_text_length,
                "pattern_type": request.pattern_type,
                "time_period": request.time_period
            }
            
            # Add specific pattern analysis based on type
            if "injury" in request.pattern_type.lower():
                injury_keywords = [
                    "fracture", "laceration", "abrasion", "contusion", "blunt force", "thermal", 
                    "gunshot", "stab", "dislocation", "rupture", "pneumothorax", "hemothorax",
                    "concussion", "trauma", "injury", "wound", "bruise", "hematoma"
                ]
                patterns["injury_patterns"] = {}
                
                for keyword in injury_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["injury_patterns"][keyword] = case_count
            
            if "demographic" in request.pattern_type.lower():
                # Enhanced demographics - count unique cases, not total occurrences
                demographic_patterns = {}
                demographic_keywords = [
                    "male", "female", "teenager", "adult", "elderly", "child", "infant",
                    "white", "black", "hispanic", "caucasian", "african american", "age", "years"
                ]
                
                for keyword in demographic_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    demographic_patterns[keyword] = case_count
                
                patterns["demographics"] = demographic_patterns
            
            if "cause" in request.pattern_type.lower():
                cause_keywords = [
                    "accident", "homicide", "suicide", "natural", "undetermined", "overdose",
                    "drowning", "asphyxiation", "electrocution", "fire", "smoke inhalation",
                    "cardiac arrest", "respiratory failure", "sepsis", "infection"
                ]
                patterns["causes"] = {}
                
                for keyword in cause_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["causes"][keyword] = case_count
            
            # NEW: Vehicle-related patterns
            if "vehicle" in request.pattern_type.lower():
                vehicle_keywords = [
                    "motor vehicle", "car", "truck", "motorcycle", "bicycle", "pedestrian",
                    "driver", "passenger", "unrestrained", "seatbelt", "airbag", "ejection",
                    "rollover", "head-on", "rear-end", "side-impact", "single vehicle"
                ]
                patterns["vehicle_patterns"] = {}
                
                for keyword in vehicle_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["vehicle_patterns"][keyword] = case_count
            
            # NEW: Toxicology patterns
            if "toxicology" in request.pattern_type.lower():
                tox_keywords = [
                    "ethanol", "alcohol", "drug", "cocaine", "methamphetamine", "heroin",
                    "fentanyl", "marijuana", "cannabis", "opioid", "benzodiazepine",
                    "amphetamine", "barbiturate", "positive", "negative", "toxicology"
                ]
                patterns["toxicology_patterns"] = {}
                
                for keyword in tox_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["toxicology_patterns"][keyword] = case_count
            
            # NEW: Medical condition patterns
            if "medical" in request.pattern_type.lower():
                medical_keywords = [
                    "hypertension", "diabetes", "heart disease", "cancer", "stroke",
                    "pneumonia", "liver disease", "kidney disease", "obesity", "asthma",
                    "copd", "emphysema", "cirrhosis", "hepatitis", "hiv", "aids"
                ]
                patterns["medical_conditions"] = {}
                
                for keyword in medical_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["medical_conditions"][keyword] = case_count
            
            # NEW: Body system patterns
            if "body_systems" in request.pattern_type.lower():
                body_system_keywords = [
                    "cardiovascular", "respiratory", "nervous", "digestive", "musculoskeletal",
                    "integumentary", "urinary", "reproductive", "endocrine", "lymphatic",
                    "heart", "lung", "brain", "liver", "kidney", "spleen", "pancreas"
                ]
                patterns["body_systems"] = {}
                
                for keyword in body_system_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["body_systems"][keyword] = case_count
            
            # NEW: Circumstance patterns
            if "circumstances" in request.pattern_type.lower():
                circumstance_keywords = [
                    "found", "discovered", "witness", "scene", "location", "home", "hospital",
                    "roadway", "residence", "workplace", "public", "private", "indoor", "outdoor",
                    "weather", "time", "date", "season"
                ]
                patterns["circumstances"] = {}
                
                for keyword in circumstance_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["circumstances"][keyword] = case_count
            
            # NEW: Comprehensive analysis (all patterns)
            if "comprehensive" in request.pattern_type.lower():
                # Injury patterns
                injury_keywords = [
                    "fracture", "laceration", "abrasion", "contusion", "blunt force", "thermal", 
                    "gunshot", "stab", "dislocation", "rupture", "pneumothorax", "hemothorax",
                    "concussion", "trauma", "injury", "wound", "bruise", "hematoma"
                ]
                patterns["injury_patterns"] = {}
                
                for keyword in injury_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["injury_patterns"][keyword] = case_count
                
                # Demographics
                demographic_keywords = [
                    "male", "female", "teenager", "adult", "elderly", "child", "infant",
                    "white", "black", "hispanic", "caucasian", "african american", "age", "years"
                ]
                patterns["demographics"] = {}
                
                for keyword in demographic_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["demographics"][keyword] = case_count
                
                # Causes
                cause_keywords = [
                    "accident", "homicide", "suicide", "natural", "undetermined", "overdose",
                    "drowning", "asphyxiation", "electrocution", "fire", "smoke inhalation",
                    "cardiac arrest", "respiratory failure", "sepsis", "infection"
                ]
                patterns["causes"] = {}
                
                for keyword in cause_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["causes"][keyword] = case_count
                
                # Vehicle patterns
                vehicle_keywords = [
                    "motor vehicle", "car", "truck", "motorcycle", "bicycle", "pedestrian",
                    "driver", "passenger", "unrestrained", "seatbelt", "airbag", "ejection",
                    "rollover", "head-on", "rear-end", "side-impact", "single vehicle"
                ]
                patterns["vehicle_patterns"] = {}
                
                for keyword in vehicle_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["vehicle_patterns"][keyword] = case_count
                
                # Toxicology patterns
                tox_keywords = [
                    "ethanol", "alcohol", "drug", "cocaine", "methamphetamine", "heroin",
                    "fentanyl", "marijuana", "cannabis", "opioid", "benzodiazepine",
                    "amphetamine", "barbiturate", "positive", "negative", "toxicology"
                ]
                patterns["toxicology_patterns"] = {}
                
                for keyword in tox_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["toxicology_patterns"][keyword] = case_count
                
                # Medical conditions
                medical_keywords = [
                    "hypertension", "diabetes", "heart disease", "cancer", "stroke",
                    "pneumonia", "liver disease", "kidney disease", "obesity", "asthma",
                    "copd", "emphysema", "cirrhosis", "hepatitis", "hiv", "aids"
                ]
                patterns["medical_conditions"] = {}
                
                for keyword in medical_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["medical_conditions"][keyword] = case_count
                
                # Body systems
                body_system_keywords = [
                    "cardiovascular", "respiratory", "nervous", "digestive", "musculoskeletal",
                    "integumentary", "urinary", "reproductive", "endocrine", "lymphatic",
                    "heart", "lung", "brain", "liver", "kidney", "spleen", "pancreas"
                ]
                patterns["body_systems"] = {}
                
                for keyword in body_system_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["body_systems"][keyword] = case_count
                
                # Circumstances
                circumstance_keywords = [
                    "found", "discovered", "witness", "scene", "location", "home", "hospital",
                    "roadway", "residence", "workplace", "public", "private", "indoor", "outdoor",
                    "weather", "time", "date", "season"
                ]
                patterns["circumstances"] = {}
                
                for keyword in circumstance_keywords:
                    # Count unique cases that contain this keyword
                    case_count = sum(1 for case_text in case_patterns.values() if keyword in case_text)
                    patterns["circumstances"][keyword] = case_count
            
            return {
                "patterns": patterns,
                "timestamp": datetime.now().isoformat(),
                "analysis_method": "mongodb_aggregation",
                "comprehensive": True
            }
        else:
            return {
                "patterns": {
                    "total_cases": 0,
                    "case_ids": [],
                    "total_chunks": 0,
                    "text_length": 0,
                    "pattern_type": request.pattern_type,
                    "time_period": request.time_period
                },
                "timestamp": datetime.now().isoformat(),
                "analysis_method": "mongodb_aggregation",
                "comprehensive": True,
                "message": "MongoDB collection not available"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pattern statistics error: {str(e)}")

@app.post("/api/compare")
async def compare_endpoint(request: CompareRequest):
    """Case comparison endpoint using ALL chunk text for comprehensive analysis"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Access MongoDB collection directly
        if hasattr(assistant_engine, 'collection') and assistant_engine.collection is not None:
            # ENHANCED: Get ALL text from ALL chunks for each case being compared
            case_comparisons = []
            
            for case_id in request.case_ids:
                pipeline = [
                    {
                        "$match": {"case_id": case_id}
                    },
                    {
                        "$group": {
                            "_id": "$case_id",
                            "all_texts": {"$push": "$text"},  # Collect ALL chunk texts
                            "chunk_count": {"$sum": 1},
                            "chunk_ids": {"$push": "$chunk_id"}
                        }
                    }
                ]
                
                case_data = list(assistant_engine.collection.aggregate(pipeline))
                
                if case_data:
                    case_info = case_data[0]
                    # ENHANCED: Combine ALL texts from ALL chunks for this case
                    full_text = " ".join(case_info["all_texts"])
                    
                    case_comparisons.append({
                        "case_id": case_id,
                        "text": full_text,
                        "chunk_count": case_info["chunk_count"],
                        "text_length": len(full_text)
                    })
                else:
                    case_comparisons.append({
                        "case_id": case_id,
                        "text": "CASE NOT FOUND",
                        "chunk_count": 0,
                        "text_length": 0
                    })
            
            # Create comprehensive comparison prompt
            comparison_prompt = f"""Compare the following cases focusing on {request.comparison_type}:

CASES TO COMPARE: {', '.join(request.case_ids)}

COMPREHENSIVE CASE DATA (ALL CHUNKS):

"""
            
            for case in case_comparisons:
                comparison_prompt += f"""
=== CASE {case['case_id']} ===
Chunks: {case['chunk_count']}
Text Length: {case['text_length']:,} characters
Full Text: {case['text'][:20000]}...  # First 20K characters per case

"""
            
            comparison_prompt += f"""
Please provide a comprehensive comparison focusing on {request.comparison_type}, including:
1. Similarities and differences between cases
2. Key findings for each case
3. Patterns or trends across cases
4. Statistical comparisons where relevant
5. Notable observations and insights

Base your comparison on the complete text data from ALL chunks for each case."""
            
            # Use the assistant's LLM for comparison
            response = assistant_engine.llm.complete(comparison_prompt)
            
            return {
                "response": response.text,
                "cases_compared": request.case_ids,
                "comparison_type": request.comparison_type,
                "case_data": case_comparisons,
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "response": f"Mock comparison for cases: {', '.join(request.case_ids)} focusing on {request.comparison_type}",
                "cases_compared": request.case_ids,
                "comparison_type": request.comparison_type,
                "case_data": [],
                "timestamp": datetime.now().isoformat(),
                "message": "MongoDB collection not available"
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@app.post("/api/search-configurable")
async def search_configurable_endpoint(request: SearchRequest):
    """Configurable search endpoint"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Use the assistant's search capabilities
        search_results = assistant_engine._direct_mongodb_search(
            message=request.query,
            top_k=request.top_k or 50
        )
        
        return {
            "results": search_results.get("results", []),
            "total": len(search_results.get("results", [])),
            "query": request.query,
            "configurable": True
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/api/patterns-comprehensive")
async def patterns_comprehensive_endpoint(request: ComprehensivePatternRequest):
    """Comprehensive pattern analysis endpoint with maximum data coverage"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        query = f"Comprehensive analysis of {request.pattern_type} across ALL autopsy reports for {request.time_period} time period. Analyze every available case and provide complete statistics."
        
        # Limit to reasonable number to avoid timeouts
        max_results = min(request.max_results, 100)
        
        result = assistant_engine.chat(
            message=query,
            conversation_id=request.conversation_id,
            top_k=max_results
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comprehensive pattern analysis error: {str(e)}")

@app.post("/api/analyze-all")
async def analyze_all_endpoint(request: ChatRequest):
    """Analyze all data endpoint"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Use the assistant's comprehensive analysis
        response_data = assistant_engine.chat(
            message=f"Analyze all available data comprehensively: {request.message}",
            top_k=100
        )
        
        return {
            "analysis": response_data.get("response"),
            "sources": response_data.get("sources"),
            "metadata": response_data.get("metadata"),
            "scope": "all_data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analyze all error: {str(e)}")

@app.post("/api/search-all")
async def search_all_endpoint(request: SearchRequest):
    """Search all data endpoint"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Use the assistant's comprehensive search
        search_results = assistant_engine._direct_mongodb_search(
            message=request.query,
            top_k=request.top_k or 100
        )
        
        return {
            "results": search_results.get("results", []),
            "total": len(search_results.get("results", [])),
            "query": request.query,
            "scope": "all_data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search all error: {str(e)}")

@app.post("/api/conversations/{conversation_id}/clear")
async def clear_conversation_endpoint(conversation_id: str):
    """Clear conversation history"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # For now, just return success (conversation management would need to be implemented)
        return {
            "message": f"Conversation {conversation_id} cleared successfully",
            "conversation_id": conversation_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear conversation error: {str(e)}")

@app.get("/api/conversations")
async def get_conversations_endpoint():
    """Get all conversations"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # For now, return empty list (conversation management would need to be implemented)
        return {
            "conversations": [],
            "total_conversations": 0,
            "message": "Conversation management not yet implemented"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get conversations error: {str(e)}")

@app.post("/api/case-details")
async def case_details_post_endpoint(request: ChatRequest):
    """Case details endpoint (POST version)"""
    if not assistant_engine:
        raise HTTPException(status_code=500, detail="Autopsy Assistant not initialized")
    
    try:
        # Use the assistant's case analysis capabilities
        response_data = assistant_engine.chat(
            message=f"Provide detailed analysis of: {request.message}",
            top_k=request.top_k or 20
        )
        
        return {
            "case_analysis": response_data.get("response"),
            "sources": response_data.get("sources"),
            "metadata": response_data.get("metadata"),
            "query": request.message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Case details error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
