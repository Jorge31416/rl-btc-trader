"""
Cliente directo para Binance Demo Futures (demo-fapi.binance.com).
Usa requests + HMAC SHA256 — sin ccxt.
"""
import hmac
import hashlib
import time
import logging
import requests

log = logging.getLogger(__name__)
BASE_URL = "https://demo-fapi.binance.com"


class BinanceDemoClient:

    def __init__(self, api_key: str, api_secret: str):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.session    = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = int(time.time() * 1000)
        query = "&".join(f"{k}={v}" for k, v in params.items())
        params["signature"] = hmac.new(
            self.api_secret.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        return params

    def _get(self, path: str, params: dict = None, signed: bool = False):
        params = params or {}
        if signed:
            params = self._sign(params)
        for attempt in range(3):
            try:
                r = self.session.get(f"{BASE_URL}{path}", params=params, timeout=15)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _post(self, path: str, params: dict = None):
        params = params or {}
        params = self._sign(params)
        r = self.session.post(f"{BASE_URL}{path}", params=params, timeout=15)
        data = r.json()
        if isinstance(data, dict) and data.get("code", 0) < 0:
            raise Exception(f"Binance error {data['code']}: {data['msg']}")
        return data

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m",
                    limit: int = 200) -> list:
        sym = symbol.replace("/", "").replace(":USDT", "")
        data = self._get("/fapi/v1/klines",
                         {"symbol": sym, "interval": timeframe, "limit": limit})
        return [[int(k[0]), float(k[1]), float(k[2]),
                 float(k[3]), float(k[4]), float(k[5])] for k in data]

    def fetch_balance(self) -> dict:
        data   = self._get("/fapi/v2/balance", signed=True)
        result = {"USDT": {"total": 0.0, "free": 0.0}}
        for asset in data:
            if asset["asset"] == "USDT":
                result["USDT"] = {
                    "total": float(asset["balance"]),
                    "free":  float(asset["availableBalance"]),
                }
        return result

    def fetch_positions(self, symbols: list = None) -> list:
        data   = self._get("/fapi/v2/positionRisk", signed=True)
        result = []
        wanted = {s.replace("/", "").replace(":USDT", "") for s in (symbols or [])}
        for pos in data:
            if wanted and pos["symbol"] not in wanted:
                continue
            amt = float(pos["positionAmt"])
            if abs(amt) > 0:
                result.append({
                    "symbol":    pos["symbol"],
                    "contracts": abs(amt),
                    "side":      "long" if amt > 0 else "short",
                    "entryPrice": float(pos["entryPrice"]),
                })
        return result

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        sym  = symbol.replace("/", "").replace(":USDT", "")
        body = {"symbol": sym, "side": side.upper(),
                "type": "MARKET", "quantity": round(amount, 3)}
        data = self._post("/fapi/v1/order", body)
        return {"id": str(data.get("orderId", "")),
                "average": float(data.get("avgPrice", 0) or 0)}
