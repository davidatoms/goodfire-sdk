import goodfire
import os
import logging
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
GOODFIRE_API_KEY = os.getenv("GOODFIRE_API_KEY")
if not GOODFIRE_API_KEY:
    raise ValueError("Please set GOODFIRE_API_KEY in your .env file")

def save_response(output_dir: Path, example_name: str, response: str, metadata: dict = None):
    """Save a response to a file in the output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{example_name}_{timestamp}.txt"
    filepath = output_dir / filename
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    # Save the response
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(response)
    
    # Save metadata if provided
    if metadata:
        metadata_file = output_dir / f"{example_name}_{timestamp}_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    return filepath

def track_feature_activations(client, messages, variant, output_dir: Path, example_name: str):
    """Track and save feature activations for a conversation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get feature activations
    context = client.features.inspect(
        messages=messages,
        model=variant
    )
    
    # Get top activating features
    top_features = context.top(k=10)
    
    # Save feature activations
    feature_file = output_dir / f"{example_name}_{timestamp}_features.json"
    with open(feature_file, 'w', encoding='utf-8') as f:
        json.dump({
            "top_features": [
                {"feature": str(feature), "activation": activation}
                for feature, activation in top_features
            ]
        }, f, indent=2)
    
    return top_features

def track_logits(client, messages, variant, output_dir: Path, example_name: str):
    """Track and save logits for a conversation."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get logits
    logits_response = client.chat.logits(
        messages=messages,
        model=variant,
        top_k=10
    )
    
    # Save logits
    logits_file = output_dir / f"{example_name}_{timestamp}_logits.json"
    with open(logits_file, 'w', encoding='utf-8') as f:
        json.dump(logits_response.logits, f, indent=2)
    
    return logits_response

def main():
    # Set up logging
    log_file = setup_logging()
    logger = logging.getLogger("goodfire")
    logger.info("Starting Goodfire SDK playground")
    
    # Set up output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir.absolute()}")
    
    try:
        # Initialize the client
        logger.info("Initializing Goodfire client")
        client = goodfire.Client(api_key=GOODFIRE_API_KEY)
        
        # Create a model variant
        logger.info("Creating model variant")
        variant = goodfire.Variant("meta-llama/Llama-3.3-70B-Instruct")
        
        # Example 1: Basic chat completion with tracking
        logger.info("Running Example 1: Basic Chat Completion")
        print("\n=== Example 1: Basic Chat Completion ===")
        messages1 = [{"role": "user", "content": "Hi, how are you?"}]
        
        # Track logits before generation
        logits1 = track_logits(client, messages1, variant, output_dir, "basic_chat_logits")
        
        response1 = ""
        for token in client.chat.completions.create(
            messages1,
            model=variant,
            stream=True,
            max_completion_tokens=100,
        ):
            content = token.choices[0].delta.content
            print(content, end="")
            response1 += content
        
        # Track feature activations after generation
        features1 = track_feature_activations(client, messages1, variant, output_dir, "basic_chat_features")
        
        # Save response with enhanced metadata
        save_response(
            output_dir,
            "basic_chat",
            response1,
            {
                "model": variant.base_model,
                "max_tokens": 100,
                "top_features": [str(f) for f, _ in features1],
                "top_logits": logits1.logits
            }
        )
        
        # Example 2: Auto steering for humor with tracking
        logger.info("Running Example 2: Auto Steering for Humor")
        print("\n\n=== Example 2: Auto Steering for Humor ===")
        variant.reset()
        edits = client.features.AutoSteer(
            specification="be funny",
            model=variant,
        )
        variant.set(edits)
        logger.debug(f"Applied edits: {edits}")
        print(f"Applied edits: {edits}")
        
        messages2 = [{"role": "user", "content": "Tell me about pirates"}]
        
        # Track logits before generation
        logits2 = track_logits(client, messages2, variant, output_dir, "auto_steer_logits")
        
        response2 = ""
        for token in client.chat.completions.create(
            messages2,
            model=variant,
            stream=True,
            max_completion_tokens=120,
        ):
            content = token.choices[0].delta.content
            print(content, end="")
            response2 += content
        
        # Track feature activations after generation
        features2 = track_feature_activations(client, messages2, variant, output_dir, "auto_steer_features")
        
        # Save response with enhanced metadata
        save_response(
            output_dir,
            "auto_steer_humor",
            response2,
            {
                "model": variant.base_model,
                "max_tokens": 120,
                "edits": str(edits),
                "top_features": [str(f) for f, _ in features2],
                "top_logits": logits2.logits
            }
        )
        
        # Example 3: Feature search and manual feature editing with tracking
        logger.info("Running Example 3: Feature Search and Manual Editing")
        print("\n\n=== Example 3: Feature Search and Manual Editing ===")
        variant.reset()
        funny_features = client.features.search(
            "funny",
            model=variant,
            top_k=3
        )
        logger.debug(f"Found features: {funny_features}")
        print(f"Found features: {funny_features}")
        
        # Set a feature weight
        variant.set(funny_features[0], 0.6)
        
        messages3 = [{"role": "user", "content": "tell me about foxes"}]
        
        # Track logits before generation
        logits3 = track_logits(client, messages3, variant, output_dir, "manual_feature_logits")
        
        response3 = ""
        for token in client.chat.completions.create(
            messages3,
            model=variant,
            stream=True,
            max_completion_tokens=100,
        ):
            content = token.choices[0].delta.content
            print(content, end="")
            response3 += content
        
        # Track feature activations after generation
        features3 = track_feature_activations(client, messages3, variant, output_dir, "manual_feature_features")
        
        # Save response with enhanced metadata
        save_response(
            output_dir,
            "manual_feature_edit",
            response3,
            {
                "model": variant.base_model,
                "max_tokens": 100,
                "features": str(funny_features[0]),
                "weight": 0.6,
                "top_features": [str(f) for f, _ in features3],
                "top_logits": logits3.logits
            }
        )
        
        # Example 4: Feature inspection with tracking
        logger.info("Running Example 4: Feature Inspection")
        print("\n\n=== Example 4: Feature Inspection ===")
        variant.reset()
        joke_conversation = [
            {"role": "user", "content": "Hello how are you?"},
            {"role": "assistant", "content": "What do you call an alligator in a vest? An investigator!"}
        ]
        
        # Track logits before inspection
        logits4 = track_logits(client, joke_conversation, variant, output_dir, "feature_inspection_logits")
        
        context = client.features.inspect(
            messages=joke_conversation,
            model=variant,
        )
        
        # Get top activating features
        top_features = context.top(k=5)
        logger.debug(f"Top activating features: {top_features}")
        print(f"Top activating features: {top_features}")
        
        # Save feature inspection results with enhanced metadata
        save_response(
            output_dir,
            "feature_inspection",
            str(top_features),
            {
                "model": variant.base_model,
                "context": "joke conversation",
                "top_logits": logits4.logits
            }
        )
        
        logger.info("Playground completed successfully")
        
    except Exception as e:
        logger.error(f"Error in playground: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 