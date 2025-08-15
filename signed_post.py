import os
import json
import requests
import boto3
from typing import Dict
from urllib.parse import urljoin
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import ReadOnlyCredentials

# === Config y helpers ===

def must_env(name: str, default: str | None = None) -> str:
    v = os.environ.get(name, default) if default is not None else os.environ.get(name)
    if not v:
        raise SystemExit(f"Define la variable de entorno {name}")
    return str(v)

AWS_PROFILE: str = os.environ.get("AWS_PROFILE", "camelscl")
AWS_REGION: str  = os.environ.get("AWS_REGION", "sa-east-1")
FN_URL: str      = must_env("FN_URL")  # p. ej. https://xxxxx.lambda-url.sa-east-1.on.aws/

# Aseguramos que termine con "/"
if not FN_URL.endswith("/"):
    FN_URL = FN_URL + "/"

# Credenciales firmadas (perfil/region)
session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
creds = session.get_credentials()
if creds is None:
    raise SystemExit(f"No encontré credenciales para el perfil '{AWS_PROFILE}'")
frozen = creds.get_frozen_credentials()
readonly = ReadOnlyCredentials(frozen.access_key, frozen.secret_key, frozen.token)

def signed_request(method: str, url: str, body: Dict | bytes | str | None = None, headers: Dict[str, str] | None = None) -> requests.Response:
    if headers is None:
        headers = {"Content-Type": "application/json"}

    data: bytes
    if body is None:
        data = b""
    elif isinstance(body, (bytes, bytearray)):
        data = bytes(body)
    elif isinstance(body, (dict, list)):
        data = json.dumps(body).encode("utf-8")
    else:
        data = str(body).encode("utf-8")

    req = AWSRequest(method=method, url=url, data=data, headers=headers)
    SigV4Auth(readonly, "lambda", AWS_REGION).add_auth(req)
    return requests.request(method, url, data=data, headers=dict(req.headers))

def main() -> None:
    # health
    r = signed_request("GET", urljoin(FN_URL, "healthz"))
    print("healthz:", r.status_code, r.text)

    # ready
    r = signed_request("GET", urljoin(FN_URL, "readyz"))
    print("readyz :", r.status_code, r.text)

    # predict
    payload = {
        "basin_id": "3434003",
        "features": {
            "prcp_mm_cr2met_lag1": 2.1,
            "pet_mm_hargreaves_lag3": 1.0,
            "deficit_sum_7d": 0.0,
            "tmean_c_lag1": 12.3,
            "swe_mm_lag1": 4.5
        }
    }
    r = signed_request("POST", urljoin(FN_URL, "predict"), body=payload)
    print("predict:", r.status_code)
    try:
        print(json.dumps(r.json(), ensure_ascii=False, indent=2))
    except Exception:
        print(r.text)

if __name__ == "__main__":
    main()
