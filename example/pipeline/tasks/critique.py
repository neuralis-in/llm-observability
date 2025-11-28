from aiobs.observe import observe

@observe(enh_prompt=True, threshold=10)
def critique(draft: str, client, model: str) -> str:
    """Critique and refine a draft answer for clarity and accuracy."""
    messages = [
        {"role": "system", "content": "You are a careful editor improving clarity and accuracy."},
        {
            "role": "user",
            "content": (
                "Critique this answer for clarity, accuracy, and completeness. "
                "Then return an improved version in 3-6 sentences.\n\n"
                f"Answer:\n{draft}"
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_tokens=250,
    )
    return (resp.choices[0].message.content or "").strip()

