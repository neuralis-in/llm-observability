"""
Simple Gemini Example with aiobs Observability

This example demonstrates how to use the aiobs library to observe
Gemini API calls using the credentials manager for GCP authentication.

Usage:
    # Set your credentials path (or use default GCP credentials)
    export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    
    # Optionally set region
    export GOOGLE_CLOUD_REGION=us-central1
    
    # Run the example
    python main.py
"""

from aiobs import observer, observe
from credentials_manager import CredentialsManager


def simple_generation():
    """Simple text generation example."""
    creds_manager = CredentialsManager('service_account_credentials.json')
    client = creds_manager.get_authenticated_client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="What is the capital of France? Answer in one sentence."
    )

    print(f"Response: {response.text}")
    return response.text

@observe(enh_prompt=True)
def generation_with_system_instruction():
    """Generation with system instruction example."""
    creds_manager = CredentialsManager()
    client = creds_manager.get_authenticated_client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="Tell me a joke",
        config={
            "system_instruction": "You are a helpful assistant that tells short, family-friendly jokes.",
            "temperature": 0.9,
            "max_output_tokens": 100,
        }
    )

    print(f"Joke: {response.text}")
    return response.text

def ask_to_remember():
    """Ask to remember example."""
    creds_manager = CredentialsManager()
    client = creds_manager.get_authenticated_client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="My name is Alice. Remember that."
    )
    print(f"Remember: {response.text}")
    return response.text


def ask_to_recall():
    """Ask to recall example."""
    creds_manager = CredentialsManager()
    client = creds_manager.get_authenticated_client()

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents="What is my name?"
    )
    print(f"Recall: {response.text}")
    return response.text

def multi_turn_conversation():
    """Multi-turn conversation example."""
    creds_manager = CredentialsManager()
    client = creds_manager.get_authenticated_client()

    # First turn
    response1 = ask_to_remember()
    print(f"Turn 1: {response1}")

    # Second turn with context
    response2 = ask_to_recall()
    print(f"Turn 2: {response2}")
    return response2


def main():
    """Run all examples with observability enabled."""
    # Start observability session
    session_id = observer.observe(session_name="gemini-example")
    print(f"Started observability session: {session_id}\n")

    try:
        print("=" * 50)
        print("1. Simple Generation")
        print("=" * 50)
        simple_generation()
        print()

        print("=" * 50)
        print("2. Generation with System Instruction")
        print("=" * 50)
        generation_with_system_instruction()
        print()

        print("=" * 50)
        print("3. Multi-turn Conversation")
        print("=" * 50)
        multi_turn_conversation()
        print()

    finally:
        # End session and flush observability data
        observer.end()
        output_path = observer.flush()
        print(f"\n✅ Observability data saved to: {output_path}")


if __name__ == "__main__":
    main()

