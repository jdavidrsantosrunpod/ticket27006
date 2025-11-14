import os
import json
import re
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    Settings,
    StorageContext,
    load_index_from_storage
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama

# LlamaIndex Advanced Features (with fallbacks)
try:
    from llama_index.core.memory import ChatMemoryBuffer
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("⚠️  ChatMemoryBuffer not available")

try:
    from llama_index.core.agent import ReActAgent
    from llama_index.core.tools import QueryEngineTool, ToolMetadata
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False
    print("⚠️  ReActAgent not available")

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("⚠️  HuggingFace embeddings not available")

# MongoDB imports
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

# Environment variables
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
# Configuration
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "hde-dev")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "autopsy-vectors")
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
LOCAL_LLM = os.getenv("LOCAL_LLM", "llama3:8b")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "50"))  # Default number of results to return

# Ollama configuration
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")

class AutopsyAssistantEngine:
    """Enhanced autopsy assistant using MongoDB data with available LlamaIndex capabilities."""
    
    def __init__(self):
        print("🚀 Initializing Enhanced Autopsy Assistant with Available LlamaIndex Capabilities...")
        
        # Initialize MongoDB connection
        self._initialize_mongodb()
        
        # Initialize local embedding model
        self._initialize_embeddings()
        
        # Initialize LLM
        self._initialize_llm()
        
        # Initialize LlamaIndex components with available features
        self._initialize_llamaindex_components()
        
        # Initialize conversation memory if available
        self.conversation_memories = {}
        
        print("✅ Enhanced LlamaIndex Assistant initialized successfully!")
    
    def _initialize_mongodb(self):
        """Initialize MongoDB connection for data retrieval."""
        try:
            self.client = MongoClient(MONGO_URI)
            self.db = self.client[DB_NAME]
            self.collection = self.db[COLLECTION_NAME]
            self.model = SentenceTransformer(MODEL_NAME)
            print("🗄️ MongoDB connection initialized")
        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            raise
    
    def _initialize_embeddings(self):
        """Initialize local embedding model."""
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                Settings.embed_model = self.embed_model
                print("🔤 Local embedding model initialized")
            except Exception as e:
                print(f"⚠️  Could not initialize HuggingFace embeddings: {e}")
                self.embed_model = None
                Settings.embed_model = None
        else:
            print("🔤 Using default embeddings")
            self.embed_model = None
            Settings.embed_model = None
    
    def _initialize_llm(self):
        """Initialize local LLM with fallback."""
        try:
            # Configure Ollama with the correct host and port
            ollama_base_url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
            print(f"🔗 Connecting to Ollama at: {ollama_base_url}")
            
            self.llm = Ollama(
                model=LOCAL_LLM, 
                request_timeout=120.0,
                base_url=ollama_base_url
            )
            print(f"🤖 Local LLM initialized: {LOCAL_LLM}")
            self.llm_available = True
        except Exception as e:
            print(f"⚠️  Could not initialize Ollama LLM: {e}")
            print("🔧 Using fallback response system")
            self.llm = None
            self.llm_available = False
    
    def _initialize_llamaindex_components(self):
        """Initialize LlamaIndex components with available features."""
        # Create MongoDB-based retriever
        self._create_mongodb_retriever()
        
        # Create response synthesizer
        self._create_response_synthesizer()
        
        # Create query engine
        self._create_query_engine()
        
        # Create agent if available
        if AGENT_AVAILABLE:
            self._create_agent()
        
        print("🔍 LlamaIndex RAG components initialized")
    
    def _create_mongodb_retriever(self):
        """Create MongoDB-based retriever for LlamaIndex."""
        class MongoDBRetriever:
            def __init__(self, collection, model, embed_model):
                self.collection = collection
                self.model = model
                self.embed_model = embed_model
            
            def retrieve(self, query: str, similarity_top_k: int = None):
                # Use default if not specified
                if similarity_top_k is None:
                    similarity_top_k = DEFAULT_TOP_K
                
                # Check if query contains a specific case ID
                import re
                case_match = re.search(r'case\s*(\d{6})', query, re.IGNORECASE)
                if case_match:
                    case_id = case_match.group(1)
                    print(f"🔍 Found case ID in query: {case_id}")
                    
                    # Search for exact case ID first
                    exact_results = list(self.collection.find({"case_id": case_id}))
                    print(f"✅ Found {len(exact_results)} exact matches for case {case_id}")
                    
                    if exact_results:
                        # Convert to LlamaIndex nodes
                        from llama_index.core.schema import TextNode
                        nodes = []
                        for result in exact_results:
                            node = TextNode(
                                text=result["text"],
                                metadata={
                                    "case_id": result.get("case_id"),
                                    "chunk_id": result.get("chunk_id"),
                                    "score": 1.0  # High score for exact matches
                                }
                            )
                            nodes.append(node)
                        return nodes
                
                # Fall back to vector search
                print(f"🔍 Using vector search for: {query}")
                query_embedding = self.model.encode(query)
                
                # Search MongoDB with increased candidates for better coverage
                pipeline = [
                    {
                        "$vectorSearch": {
                            "queryVector": query_embedding.tolist(),
                            "path": "embedding",
                            "numCandidates": similarity_top_k * 20,  # Increased from 10 to 20
                            "limit": similarity_top_k,
                            "index": "default"
                        }
                    },
                    {
                        "$project": {
                            "text": 1,
                            "case_id": 1,
                            "chunk_id": 1,
                            "score": {"$meta": "vectorSearchScore"}
                        }
                    }
                ]
                
                results = list(self.collection.aggregate(pipeline))
                print(f"🔍 Vector search returned {len(results)} results")
                
                # Convert to LlamaIndex nodes
                from llama_index.core.schema import TextNode
                nodes = []
                for result in results:
                    node = TextNode(
                        text=result["text"],
                        metadata={
                            "case_id": result.get("case_id"),
                            "chunk_id": result.get("chunk_id"),
                            "score": result.get("score", 0)
                        }
                    )
                    nodes.append(node)
                
                return nodes
        
        self.retriever = MongoDBRetriever(self.collection, self.model, self.embed_model)
        print("🔍 MongoDB retriever created")
    
    def _create_response_synthesizer(self):
        """Create response synthesizer."""
        if self.llm_available and self.llm:
            self.response_synthesizer = get_response_synthesizer(
                response_mode="compact",
                llm=self.llm
            )
            print("🔧 Response synthesizer created")
        else:
            self.response_synthesizer = None
            print("🔧 Response synthesizer skipped (LLM not available)")
    
    def _create_query_engine(self):
        """Create query engine that works with MongoDB retriever."""
        try:
            # Create a simple query engine that uses our working MongoDB retriever
            from llama_index.core.query_engine import CustomQueryEngine
            
            class MongoDBQueryEngine(CustomQueryEngine):
                def __init__(self, retriever, llm, llm_available=True):
                    self.retriever = retriever
                    self.llm = llm
                    self.llm_available = llm_available
                
                def custom_query(self, query_str: str):
                    """Required abstract method."""
                    return self.query(query_str)
                
                def query(self, query_str: str):
                    print(f"🔍 MongoDB Query Engine: Searching for '{query_str}'")
                    
                    # Get nodes from MongoDB retriever with increased results
                    nodes = self.retriever.retrieve(query_str, similarity_top_k=DEFAULT_TOP_K)
                    print(f"✅ Retrieved {len(nodes)} nodes from MongoDB")
                    
                    if not nodes:
                        # Fallback response if no nodes found
                        response_text = f"I searched the autopsy reports for '{query_str}' but found no relevant cases. Please try a different search term."
                        return type('Response', (), {'response': response_text})()
                    
                    # Create context from retrieved nodes
                    context_parts = []
                    for i, node in enumerate(nodes, 1):
                        case_id = node.metadata.get('case_id', 'Unknown')
                        text = node.text.strip()
                        context_parts.append(f"=== CASE {case_id} ===\n{text}\n")
                    
                    context = "\n".join(context_parts)
                    
                    # Create prompt with context
                    prompt = f"""Based on the following autopsy report excerpts, answer this question: {query_str}

AUTOPSY REPORT EXCERPTS:
{context}

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information provided in these excerpts
2. Always mention specific case IDs when referencing cases (e.g., "Case 231225", "Case 242827")
3. If a specific case is mentioned in the excerpts, provide details about it
4. If the information is not available in the excerpts, say so clearly
5. Be specific about what you found and what you didn't find
6. Since you have access to more comprehensive data, provide a thorough analysis

Please provide a detailed answer:"""

                    # Get response from LLM or use fallback
                    if self.llm_available and self.llm:
                        print("🤖 Getting LLM response...")
                        response = self.llm.complete(prompt)
                        response_text = response.text if hasattr(response, 'text') else str(response)
                        print(f"✅ LLM response: {response_text[:200]}...")
                    else:
                        print("🔧 Using fallback response system")
                        # Create a simple summary response
                        case_ids = list(set([node.metadata.get('case_id', 'Unknown') for node in nodes]))
                        response_text = f"Found {len(nodes)} relevant documents from {len(case_ids)} cases: {', '.join(case_ids[:10])}. "
                        response_text += f"The search query '{query_str}' returned autopsy report excerpts. "
                        response_text += "For detailed analysis, please review the specific case documents."
                    
                    return type('Response', (), {'response': response_text})()
            
            self.query_engine = MongoDBQueryEngine(self.retriever, self.llm, self.llm_available)
            print("🔍 Custom MongoDB query engine created")
            
        except Exception as e:
            print(f"❌ Could not create query engine: {e}")
            # Create a fallback query engine
            self.query_engine = None
    
    def _create_agent(self):
        """Create LlamaIndex agent with tools if available."""
        if not AGENT_AVAILABLE:
            print("⚠️  Agent not available, skipping agent creation")
            return
        
        try:
            # Create tools for the agent
            tools = self._create_tools()
            
            # Create agent
            self.agent = ReActAgent.from_tools(
                tools,
                llm=self.llm,
                verbose=True,
                max_iterations=3
            )
            
            print("🤖 LlamaIndex agent with tools created")
        except Exception as e:
            print(f"⚠️  Could not create agent: {e}")
    
    def _create_tools(self):
        """Create tools for the agent."""
        if not AGENT_AVAILABLE:
            return []
        
        tools = []
        
        # Search tool
        search_tool = QueryEngineTool(
            query_engine=self.query_engine,
            metadata=ToolMetadata(
                name="search_autopsy_reports",
                description="Search autopsy reports for specific information"
            )
        )
        tools.append(search_tool)
        
        return tools
    
    def chat(self, message: str, conversation_id: str = None, top_k: int = None) -> dict:
        """Enhanced chat method with LlamaIndex capabilities."""
        try:
            print(f"🔍 Starting chat with message: {message}")
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid4())
            
            # Use default top_k if not specified
            if top_k is None:
                top_k = DEFAULT_TOP_K
            
            # Test retriever first
            print("🔍 Testing MongoDB retriever...")
            test_nodes = self.retriever.retrieve(message, similarity_top_k=5)
            print(f"✅ Retriever returned {len(test_nodes)} nodes")
            if test_nodes:
                print(f"📄 First node text: {test_nodes[0].text[:100]}...")
            
            # Try query engine first if available
            if self.query_engine:
                try:
                    print("🔍 Using query engine for direct search...")
                    response = self.query_engine.query(message)
                    response_text = response.response if hasattr(response, 'response') else str(response)
                    reasoning = "LlamaIndex query engine"
                    
                    # Enhance response with metadata
                    enhanced_response = self._enhance_response_with_metadata(
                        response_text, test_nodes, message, conversation_id
                    )
                    
                    return enhanced_response
                    
                except Exception as e:
                    print(f"❌ Query engine failed: {e}")
                    # Fallback to direct MongoDB search
                    response_text = self._direct_mongodb_search(message, conversation_id, top_k)
                    reasoning = "Direct MongoDB search (query engine failed)"
                    
                    if isinstance(response_text, dict):
                        return response_text
                    else:
                        # Legacy string response - enhance it
                        enhanced_response = self._enhance_response_with_metadata(
                            response_text, test_nodes, message, conversation_id
                        )
                        return enhanced_response
            else:
                # Fallback to direct MongoDB search
                print("🔍 Query engine not available, using direct MongoDB search...")
                response_text = self._direct_mongodb_search(message, conversation_id, top_k)
                reasoning = "Direct MongoDB search with LLM synthesis"
                
                if isinstance(response_text, dict):
                    return response_text
                else:
                    # Legacy string response - enhance it
                    enhanced_response = self._enhance_response_with_metadata(
                        response_text, test_nodes, message, conversation_id
                    )
                    return enhanced_response
                    
        except Exception as e:
            print(f"❌ Chat error: {e}")
            # Generate conversation ID for error response
            if not conversation_id:
                conversation_id = str(uuid4())
            
            return {
                "response": f"Error processing your request: {str(e)}",
                "references": [],
                "reasoning": "Error during chat processing",
                "conversation_id": conversation_id,
                "confidence": 0.0,
                "llm_used": "local_ollama",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _is_complex_query(self, message: str) -> bool:
        """Determine if a query requires complex reasoning."""
        complex_keywords = [
            "compare", "analyze", "pattern", "trend", "correlation", "relationship",
            "why", "how", "what if", "implications", "conclusion", "summary"
        ]
        return any(keyword in message.lower() for keyword in complex_keywords)
    
    def _extract_references(self, response_text: str) -> List[Dict]:
        """Extract case references from the response."""
        references = []
        try:
            # Look for case IDs in the response
            case_patterns = [
                r'case[:\s]*(\d{6})',  # Case followed by exactly 6 digits
                r'case\s+id[:\s]*(\d{6})',  # Case ID followed by exactly 6 digits
                r'#(\d{6})',  # Hash followed by exactly 6 digits
                r'(\d{6})',  # Any 6-digit number (primary pattern)
            ]
            
            for pattern in case_patterns:
                matches = re.findall(pattern, response_text, re.IGNORECASE)
                for case_id in matches:
                    if (len(case_id) >= 5 and case_id.isdigit() and 
                        case_id not in ['2023', '2024', '2025', '100000', '200000', '300000', '400000', '500000'] and
                        not case_id.startswith('202')):
                        references.append({
                            "case_id": case_id,
                            "type": "autopsy_report",
                            "relevance": "direct_reference"
                        })
            
            # Remove duplicates while preserving order
            seen = set()
            unique_references = []
            for ref in references:
                if ref["case_id"] not in seen:
                    seen.add(ref["case_id"])
                    unique_references.append(ref)
            
            return unique_references
        except Exception as e:
            print(f"Error extracting references: {e}")
            return []
    
    def _calculate_confidence(self, response_text: str) -> float:
        """Calculate confidence score for the response."""
        try:
            # Simple confidence calculation based on response length and content
            confidence = min(0.9, 0.5 + (len(response_text) / 1000) * 0.4)
            
            # Boost confidence if references are found
            if self._extract_references(response_text):
                confidence += 0.1
            
            return round(confidence, 2)
        except:
            return 0.5
    
    def _fallback_llm_query(self, message: str, conversation_id: str) -> Dict:
        """Fallback to direct LLM if advanced features fail."""
        try:
            prompt = f"""You are a forensic autopsy assistant. Answer this question: {message}

Please provide a comprehensive answer based on your knowledge of autopsy reports and forensic analysis. If you need specific case information, mention that you would need access to the actual autopsy reports."""

            response = self.llm.complete(prompt)
            
            return {
                "response": response.text if hasattr(response, 'text') else str(response),
                "references": [],
                "reasoning": "Direct LLM response (advanced features unavailable)",
                "conversation_id": conversation_id,
                "confidence": 0.5,
                "llm_used": "local_ollama",
                "conversation_history": [],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "response": f"Error processing query: {str(e)}",
                "references": [],
                "reasoning": "Error in query processing",
                "conversation_id": conversation_id,
                "confidence": 0.0,
                "conversation_history": [],
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }

    def _direct_mongodb_search(self, message: str, conversation_id: str = None, top_k: int = None):
        """Direct MongoDB search with LLM synthesis."""
        try:
            print(f"🔍 Direct MongoDB search for: {message}")
            
            # Get nodes from retriever
            nodes = self.retriever.retrieve(message, similarity_top_k=top_k)
            print(f"✅ Retrieved {len(nodes)} nodes from MongoDB")
            
            if not nodes:
                return {
                    "response": f"I searched the autopsy reports for '{message}' but found no relevant cases. Please try a different search term.",
                    "references": [],
                    "reasoning": "No relevant cases found in database",
                    "conversation_id": conversation_id,
                    "confidence": 0.0,
                    "llm_used": "local_ollama",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Create context from retrieved nodes
            context_parts = []
            for i, node in enumerate(nodes, 1):
                case_id = node.metadata.get('case_id', 'Unknown')
                text = node.text.strip()
                context_parts.append(f"=== CASE {case_id} ===\n{text}\n")
            
            context = "\n".join(context_parts)
            
            # Create prompt with context
            prompt = f"""Based on the following autopsy report excerpts, answer this question: {message}

AUTOPSY REPORT EXCERPTS:
{context}

IMPORTANT INSTRUCTIONS:
1. Answer based ONLY on the information provided in these excerpts
2. Always mention specific case IDs when referencing cases (e.g., "Case 231225", "Case 242827")
3. If a specific case is mentioned in the excerpts, provide details about it
4. If the information is not available in the excerpts, say so clearly
5. Be specific about what you found and what you didn't find
6. Since you have access to more comprehensive data, provide a thorough analysis
7. Count and list all relevant cases you find
8. For pattern analysis: Provide complete statistics, demographics breakdown, and comprehensive findings
9. For "all time" analysis: Ensure you analyze every case mentioned and provide complete coverage

Please provide a detailed answer:"""

            # Get response from LLM
            print("🤖 Getting LLM response...")
            response = self.llm.complete(prompt)
            response_text = response.text if hasattr(response, 'text') else str(response)
            
            print(f"✅ Direct MongoDB search response: {response_text[:200]}...")
            
            # Enhance response with metadata
            enhanced_response = self._enhance_response_with_metadata(
                response_text, nodes, message, conversation_id
            )
            
            return enhanced_response
            
        except Exception as e:
            print(f"❌ Direct MongoDB search error: {e}")
            return {
                "response": f"Error processing your request: {str(e)}",
                "references": [],
                "reasoning": "Error during search",
                "conversation_id": conversation_id,
                "confidence": 0.0,
                "llm_used": "local_ollama",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _enhance_response_with_metadata(self, response_text: str, nodes: list, message: str, conversation_id: str = None) -> dict:
        """Enhance the response with additional metadata and structured data."""
        try:
            # Extract case IDs from response
            references = self._extract_references(response_text)
            
            # Create enhanced metadata
            source_nodes = []
            for node in nodes:
                source_nodes.append({
                    "case_id": node.metadata.get('case_id', 'Unknown'),
                    "chunk_id": node.metadata.get('chunk_id', 'Unknown'),
                    "score": node.metadata.get('score', 0.0),
                    "text_preview": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                    "section": node.metadata.get('section', 'Unknown')
                })
            
            # Analyze response for structured data
            structured_data = self._extract_structured_data(response_text, nodes)
            
            # Calculate additional metrics
            confidence = self._calculate_confidence(response_text)
            response_length = len(response_text)
            num_sources = len(nodes)
            num_cases = len(set(node.metadata.get('case_id') for node in nodes))
            
            # Create enhanced response
            enhanced_response = {
                "response": response_text,
                "references": references,
                "reasoning": "Direct MongoDB search with LLM synthesis",
                "conversation_id": conversation_id,
                "confidence": confidence,
                "llm_used": "local_ollama",
                "timestamp": datetime.now().isoformat(),
                
                # Enhanced metadata
                "metadata": {
                    "query": message,
                    "response_length": response_length,
                    "num_sources": num_sources,
                    "num_cases": num_cases,
                    "search_method": "hybrid_mongodb_llamaindex"
                },
                
                # Source information
                "sources": source_nodes,
                
                # Structured data
                "structured_data": structured_data,
                
                # Analysis metrics
                "analysis": {
                    "has_specific_cases": len(references) > 0,
                    "has_medical_details": self._has_medical_terms(response_text),
                    "has_demographics": self._has_demographics(response_text),
                    "has_toxicology": self._has_toxicology(response_text),
                    "has_injuries": self._has_injury_terms(response_text)
                }
            }
            
            return enhanced_response
            
        except Exception as e:
            print(f"❌ Error enhancing response: {e}")
            return {
                "response": response_text,
                "references": self._extract_references(response_text),
                "reasoning": "Direct MongoDB search with LLM synthesis",
                "conversation_id": conversation_id,
                "confidence": 0.5,
                "llm_used": "local_ollama",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _extract_structured_data(self, response_text: str, nodes: list) -> dict:
        """Extract structured data from the response and nodes."""
        try:
            structured_data = {
                "cases": [],
                "demographics": {},
                "injuries": [],
                "causes_of_death": [],
                "toxicology_findings": [],
                "statistics": {}
            }
            
            # Extract case information
            import re
            case_pattern = r'Case\s+(\d{6})[:\s]*([^.\n]+)'
            case_matches = re.findall(case_pattern, response_text, re.IGNORECASE)
            
            for case_id, description in case_matches:
                case_info = {
                    "case_id": case_id,
                    "description": description.strip(),
                    "source_nodes": [n.metadata.get('chunk_id') for n in nodes if n.metadata.get('case_id') == case_id]
                }
                structured_data["cases"].append(case_info)
            
            # Extract demographics
            age_pattern = r'(\d+)\s*(?:year|yr)\s*(?:old|of age)'
            ages = re.findall(age_pattern, response_text, re.IGNORECASE)
            if ages:
                structured_data["demographics"]["ages"] = [int(age) for age in ages]
            
            gender_pattern = r'(male|female)'
            genders = re.findall(gender_pattern, response_text, re.IGNORECASE)
            if genders:
                structured_data["demographics"]["genders"] = list(set(genders))
            
            # Extract injuries
            injury_terms = [
                "fracture", "laceration", "contusion", "abrasion", "dislocation",
                "rupture", "hemorrhage", "pneumothorax", "axonal injury"
            ]
            for term in injury_terms:
                if term.lower() in response_text.lower():
                    structured_data["injuries"].append(term)
            
            # Extract causes of death
            death_pattern = r'death.*?(?:due to|caused by|result of)\s*([^.\n]+)'
            death_matches = re.findall(death_pattern, response_text, re.IGNORECASE)
            structured_data["causes_of_death"] = [match.strip() for match in death_matches]
            
            # Extract toxicology
            tox_pattern = r'(?:ethanol|alcohol|cannabis|drug).*?(?:detected|found|present)'
            tox_matches = re.findall(tox_pattern, response_text, re.IGNORECASE)
            structured_data["toxicology_findings"] = tox_matches
            
            # Calculate statistics
            structured_data["statistics"] = {
                "total_cases": len(structured_data["cases"]),
                "age_range": {
                    "min": min(structured_data["demographics"].get("ages", [0])),
                    "max": max(structured_data["demographics"].get("ages", [0]))
                } if structured_data["demographics"].get("ages") else None,
                "injury_types": len(structured_data["injuries"]),
                "has_toxicology": len(structured_data["toxicology_findings"]) > 0
            }
            
            return structured_data
            
        except Exception as e:
            print(f"❌ Error extracting structured data: {e}")
            return {}
    
    def _has_medical_terms(self, text: str) -> bool:
        """Check if text contains medical terminology."""
        medical_terms = [
            "autopsy", "fracture", "laceration", "contusion", "hemorrhage",
            "pneumothorax", "aorta", "spine", "skull", "brain", "liver"
        ]
        return any(term.lower() in text.lower() for term in medical_terms)
    
    def _has_demographics(self, text: str) -> bool:
        """Check if text contains demographic information."""
        demo_terms = ["year old", "male", "female", "age", "gender"]
        return any(term.lower() in text.lower() for term in demo_terms)
    
    def _has_toxicology(self, text: str) -> bool:
        """Check if text contains toxicology information."""
        tox_terms = ["ethanol", "alcohol", "cannabis", "drug", "toxicology"]
        return any(term.lower() in text.lower() for term in tox_terms)
    
    def _has_injury_terms(self, text: str) -> bool:
        """Check if text contains injury information."""
        injury_terms = ["fracture", "laceration", "contusion", "abrasion", "injury"]
        return any(term.lower() in text.lower() for term in injury_terms)

# Initialize the enhanced assistant
assistant_engine = AutopsyAssistantEngine()
