from __future__ import annotations


def call_api(prompt: str, options: dict, context: dict) -> dict[str, str]:
    config = options.get("config", {})
    style = config.get("style", "conservative")
    audience = context.get("vars", {}).get("audience", "reviewer")

    if style == "concise":
        output = f"[concise][{audience}] review_required: {prompt}"
    else:
        output = (
            f"[conservative][{audience}] review_required: "
            "回答前に原文、承認状態、根拠範囲を確認します。 "
            f"request={prompt}"
        )

    return {"output": output}
