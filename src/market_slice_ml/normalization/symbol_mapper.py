"""Provider-specific symbol translations."""


def to_stooq_symbol(symbol: str) -> str:
    if symbol.startswith("^") or symbol.endswith("=F"):
        return symbol
    return f"{symbol}.US"


def canonical_symbol(symbol: str) -> str:
    return symbol.upper().removesuffix(".US")
