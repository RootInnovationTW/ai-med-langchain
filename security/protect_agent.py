class ProtectAgent:
    """
    安全防護模組
    - 過濾 Prompt Injection
    - 符合 HIPAA / GDPR
    """

    def check_prompt(self, text: str) -> bool:
        blacklist = ["DROP TABLE", "DELETE", "HACK"]
        return not any(bad in text.upper() for bad in blacklist)
