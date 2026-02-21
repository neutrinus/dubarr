from infrastructure.tts_client import XTTSClient


def test_text_sanitization():
    client = XTTSClient()

    # Test cases for problematic characters
    assert client.sanitize_text("Tekst…") == "Tekst..."
    assert client.sanitize_text("— Myślnik") == "- Myślnik"
    assert client.sanitize_text("„Cytat”") == '"Cytat"'
    assert client.sanitize_text("   ") == "..."
    assert client.sanitize_text("") == "..."
    assert client.sanitize_text(None) == "..."


if __name__ == "__main__":
    # Manual run check
    client = XTTSClient()
    print(f"Sanitized '…': {client.sanitize_text('…')}")
