"""
heuristics.py
=============
Rule-based heuristic checks on a URL string.
These catch obvious phishing patterns that the ML model
(trained on the PhiUSIIL dataset distributions) may miss.
"""

import re
import ipaddress
import urllib.parse

# TLDs that are heavily abused for phishing / spam
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq",          # Freenom free TLDs, heavily abused
    "xyz", "top", "click", "link", "work",
    "loan", "win", "download", "bid",
    "stream", "gdn", "racing", "ren",
    "pw", "men", "date",
}

# Major brand names that are commonly impersonated
IMPERSONATED_BRANDS = {
    "paypal", "amazon", "apple", "google", "microsoft", "facebook",
    "instagram", "twitter", "netflix", "ebay", "bank", "wellsfargo",
    "chase", "citibank", "hsbc", "barclays", "bankofamerica",
    "linkedin", "dropbox", "icloud", "outlook", "office365",
    "steam", "valorant", "walmart", "fedex", "ups", "dhl",
}

# Keywords that strongly suggest phishing content
PHISHING_KEYWORDS = {
    "login", "signin", "sign-in", "verify", "verification",
    "update", "confirm", "account", "password", "credential",
    "secure", "security", "alert", "bank", "urgent", "suspend",
    "locked", "unlock", "reset", "recover", "restore", "validate",
}


def analyse(url: str) -> dict:
    """
    Returns a dict with:
        - score (float 0-1): how suspicious we think the URL is
        - flags (list of str): human-readable descriptions of each rule that fired
        - override (bool): True if we are very confident this is phishing
    """
    raw = url.strip()
    if not raw.startswith(("http://", "https://")):
        raw = "http://" + raw

    parsed = urllib.parse.urlparse(raw)
    domain = parsed.netloc.lower().split(":")[0]
    path   = (parsed.path + "?" + parsed.query).lower()
    full   = raw.lower()

    parts = domain.split(".")
    tld   = parts[-1] if len(parts) >= 2 else ""
    sld   = parts[-2] if len(parts) >= 2 else domain

    flags  = []
    score  = 0.0
    override = False

    # --- Rule 1: IP address domain -------------------------------------------
    try:
        ipaddress.ip_address(domain)
        flags.append("Domain is a raw IP address")
        score += 0.5
        override = True
    except ValueError:
        pass

    # --- Rule 2: Suspicious TLD ----------------------------------------------
    if tld in SUSPICIOUS_TLDS:
        flags.append(f"Suspicious free/abused TLD: .{tld}")
        score += 0.3

    # --- Rule 3: @ in URL (redirects to different host) ----------------------
    if "@" in full:
        flags.append("@ symbol in URL (possible redirect trick)")
        score += 0.4
        override = True

    # --- Rule 4: Excessive hyphens in SLD (e.g. paypal-secure-update-login) --
    hyphen_count = sld.count("-")
    if hyphen_count >= 3:
        flags.append(f"Excessive hyphens in domain ({hyphen_count})")
        score += min(0.3, hyphen_count * 0.08)

    # --- Rule 5: Brand name anywhere in the full domain ----------------------
    brand_hit = None
    for brand in IMPERSONATED_BRANDS:
        # Check every label in the domain, not just SLD
        if any(brand in part for part in parts):
            # Only flag if the domain is NOT the brand's own legitimate domain
            # (e.g. paypal.com itself is fine, paypal-login.fakesite.com is not)
            if sld != brand:
                brand_hit = brand
                flags.append(f"Brand impersonation detected: '{brand}' in domain")
                score += 0.4
                break

    # --- Rule 6: Brand name only in path/query (e.g. evil.com/paypal-login) --
    if brand_hit is None:
        for brand in IMPERSONATED_BRANDS:
            if brand in path:
                flags.append(f"Brand keyword '{brand}' in path (possible impersonation)")
                score += 0.2
                break

    # --- Override when brand impersonation + suspicious TLD both fire ---------
    if brand_hit and tld in SUSPICIOUS_TLDS:
        flags.append(f"Brand + suspicious TLD combination (high risk)")
        score = max(score, 0.9)
        override = True

    # --- Rule 7: Multiple phishing keywords in URL ---------------------------
    kw_hits = [k for k in PHISHING_KEYWORDS if k in full]
    if len(kw_hits) >= 2:
        flags.append(f"Multiple phishing keywords: {', '.join(kw_hits[:4])}")
        score += min(0.3, len(kw_hits) * 0.08)

    # --- Rule 8: Excessive subdomains (>3) -----------------------------------
    n_sub = max(0, len(parts) - 2)
    if n_sub >= 4:
        flags.append(f"Excessive subdomains ({n_sub})")
        score += 0.2

    # --- Rule 9: Unusually long domain name ----------------------------------
    if len(domain) > 50:
        flags.append(f"Unusually long domain ({len(domain)} chars)")
        score += 0.15

    # --- Rule 10: Double slash mid-URL after scheme --------------------------
    path_part = parsed.path
    if "//" in path_part:
        flags.append("Double slash in path (possible redirect)")
        score += 0.2

    score = min(score, 1.0)
    return {"score": score, "flags": flags, "override": override}
