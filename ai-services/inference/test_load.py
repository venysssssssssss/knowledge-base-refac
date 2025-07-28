"""
Test script to verify Mistral 7B model loading
Run this before starting the full service
"""

import sys
import time
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if Mistral model can be loaded successfully"""
    
    try:
        from vllm import LLM, SamplingParams
        
        logger.info("🔄 Testing Mistral 7B model loading...")
        
        # Model path (adjust if needed)
        model_path = "models/mistral-7b-instruct-v0.2"
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"❌ Model not found at {model_path}")
            logger.info("Run the download script first: ./scripts/download_model.sh")
            return False
        
        logger.info(f"📂 Model found at: {model_path}")
        
        # Initialize LLM with AWQ quantization
        logger.info("🚀 Initializing vLLM engine...")
        llm = LLM(
            model=model_path,
            quantization="awq",
            gpu_memory_utilization=0.8,  # Conservative for testing
            max_model_len=2048,  # Smaller for testing
            trust_remote_code=True
        )
        
        logger.info("✅ Model loaded successfully!")
        
        # Test generation
        logger.info("🧪 Testing generation...")
        
        sampling_params = SamplingParams(
            temperature=0.7,
            max_tokens=100,
            stop=["</s>"]
        )
        
        test_prompt = "<s>[INST] Olá! Como você pode me ajudar? [/INST]"
        
        start_time = time.time()
        outputs = llm.generate([test_prompt], sampling_params)
        generation_time = time.time() - start_time
        
        for output in outputs:
            generated_text = output.outputs[0].text
            logger.info(f"🤖 Generated response: {generated_text}")
            logger.info(f"⏱️  Generation time: {generation_time:.2f}s")
        
        logger.info("🎉 All tests passed! Model is ready for production.")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Install dependencies: pip install -r ai-services/requirements.txt")
        return False
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are installed"""
    
    logger.info("🔍 Checking dependencies...")
    
    required_packages = [
        'vllm',
        'torch',
        'transformers',
        'fastapi',
        'qdrant_client'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package}")
        except ImportError:
            logger.error(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.info("Install with: pip install -r ai-services/requirements.txt")
        return False
    
    logger.info("✅ All dependencies installed!")
    return True

if __name__ == "__main__":
    logger.info("🚀 Starting Mistral 7B test suite...")
    
    # Test dependencies first
    if not test_dependencies():
        sys.exit(1)
    
    # Test model loading
    if not test_model_loading():
        sys.exit(1)
    
    logger.info("🎯 Ready to proceed with full deployment!")
    logger.info("Next: docker compose up -d")
