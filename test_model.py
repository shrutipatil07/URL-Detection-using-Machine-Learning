import joblib
from feature_extractor import FeatureExtractor

payload = joblib.load('model.pkl')
model   = payload['model']
feat_names = payload['meta']['feature_names']
ext     = FeatureExtractor()

tests = [
    ('https://google.com',                                    'Legit'),
    ('https://github.com/openai/gpt',                         'Legit'),
    ('https://www.amazon.com/products?id=123&ref=home',        'Legit'),
    ('http://192.168.1.1/login',                               'Phishing'),
    ('http://paypal-update-account.fakesite.xyz/verify',       'Phishing'),
    ('https://secure-bankofamerica-login.suspicious.com/auth', 'Phishing'),
    ('http://mail-google-com.tk/reset-password',               'Phishing'),
]

print('URL'.ljust(55), 'Expected'.ljust(12), 'Predicted'.ljust(12), 'Conf    Status')
print('-' * 85)
for url, expected in tests:
    feats = ext.extract_features(url)
    feats = feats[[c for c in feat_names if c in feats.columns]]
    pred  = model.predict(feats)[0]
    prob  = model.predict_proba(feats)[0]
    label = 'Legit' if pred == 0 else 'Phishing'
    conf  = max(prob) * 100
    ok    = 'OK' if label == expected else 'WRONG'
    print(url[:54].ljust(55), expected.ljust(12), label.ljust(12), f'{conf:.1f}%  [{ok}]')
