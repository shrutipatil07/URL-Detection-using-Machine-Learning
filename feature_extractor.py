"""
feature_extractor.py
====================
Extracts 18 URL-only features whose computation EXACTLY matches
the corresponding columns in the PhiUSIIL dataset CSV.

No ambiguous derived metrics (CharContinuationRate, TLDLegitimateProb,
URLCharProb) are included — those cannot be faithfully replicated
from a raw URL string at runtime.
"""

import re
import ipaddress
import urllib.parse

import pandas as pd


class FeatureExtractor:
    """
    Extracts 18 lexical features from a raw URL string.
    All features are simple counts or ratios computable directly from the URL.
    They match the column names in PhiUSIIL_Phishing_URL_Dataset.csv exactly.
    """

    FEATURE_NAMES = [
        "URLLength",
        "DomainLength",
        "IsDomainIP",
        "TLDLength",
        "NoOfSubDomain",
        "HasObfuscation",
        "NoOfObfuscatedChar",
        "ObfuscationRatio",
        "NoOfLettersInURL",
        "LetterRatioInURL",
        "NoOfDegitsInURL",
        "DegitRatioInURL",
        "NoOfEqualsInURL",
        "NoOfQMarkInURL",
        "NoOfAmpersandInURL",
        "NoOfOtherSpecialCharsInURL",
        "SpacialCharRatioInURL",
        "IsHTTPS",
    ]

    def extract_features(self, url: str) -> pd.DataFrame:
        raw = url.strip()
        if not raw.startswith(("http://", "https://")):
            raw = "http://" + raw

        parsed = urllib.parse.urlparse(raw)
        domain = parsed.netloc.split(":")[0]   # strip port

        # 1. URLLength
        url_length = len(raw)

        # 2. DomainLength
        domain_length = len(domain)

        # 3. IsDomainIP
        is_ip = 0
        try:
            ipaddress.ip_address(domain)
            is_ip = 1
        except ValueError:
            pass

        # 4. TLDLength
        parts = domain.split(".")
        tld = "" if is_ip or len(parts) < 2 else parts[-1]
        tld_length = len(tld)

        # 5. NoOfSubDomain
        # domain = [sub...].SLD.TLD  =>  len(parts) - 2 subdomains
        if is_ip or len(parts) <= 2:
            no_subdomain = 0
        else:
            no_subdomain = len(parts) - 2

        # 6-8. Obfuscation: percent-encoded sequences (%xx) and @ symbols
        obfuscated = re.findall(r"%[0-9a-fA-F]{2}|@", raw)
        no_obfuscated = len(obfuscated)
        has_obfuscation = 1 if no_obfuscated > 0 else 0
        obfuscation_ratio = no_obfuscated / url_length if url_length else 0.0

        # 9-10. Letters
        no_letters = sum(c.isalpha() for c in raw)
        letter_ratio = no_letters / url_length if url_length else 0.0

        # 11-12. Digits
        no_digits = sum(c.isdigit() for c in raw)
        digit_ratio = no_digits / url_length if url_length else 0.0

        # 13. NoOfEqualsInURL
        no_equals = raw.count("=")

        # 14. NoOfQMarkInURL
        no_qmark = raw.count("?")

        # 15. NoOfAmpersandInURL
        no_amp = raw.count("&")

        # 16-17. Other special characters
        # "Special" = not alphanumeric, not [. / : _ - ]
        other_special_chars = re.findall(r"[^a-zA-Z0-9\./:_\-]", raw)
        no_other_special = len(other_special_chars)
        special_ratio = no_other_special / url_length if url_length else 0.0

        # 18. IsHTTPS
        is_https = 1 if parsed.scheme == "https" else 0

        row = [
            url_length, domain_length, is_ip, tld_length,
            no_subdomain, has_obfuscation, no_obfuscated, obfuscation_ratio,
            no_letters, letter_ratio, no_digits, digit_ratio,
            no_equals, no_qmark, no_amp, no_other_special, special_ratio,
            is_https,
        ]
        return pd.DataFrame([row], columns=self.FEATURE_NAMES)
