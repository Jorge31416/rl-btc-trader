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
        self.api_key     = api_key
        self.api_secret  = api_secret
        self.session     = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        self._time_offset = 0   # ms de diferencia local vs servidor
        self._sync_time()

    # ── Sincronización de reloj ───────────────────────────────────────────────

    def _sync_time(self):
        """Calcula el offset entre el reloj local y el servidor de Binance."""
        try:
            r           = self.session.get(f"{BASE_URL}/fapi/v1/time", timeout=5)
            server_ms   = r.json()["serverTime"]
            local_ms    = int(time.time() * 1000)
            self._time_offset = server_ms - local_ms
            if abs(self._time_offset) > 500:
                log.info(f"Offset de reloj sincronizado: {self._time_offset:+d} ms")
        except Exception as e:
            log.warning(f"No se pudo sincronizar reloj con Binance: {e}")
            self._time_offset = 0

    def _server_time(self) -> int:
        """Timestamp ajustado al reloj del servidor."""
        return int(time.time() * 1000) + self._time_offset

    # ── Firma HMAC ────────────────────────────────────────────────────────────

    def _sign(self, params: dict) -> dict:
        params["timestamp"] = self._server_time()
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
            except requests.HTTPError as e:
                # -1021 = timestamp out of sync → resincronizar y reintentar
                try:
                    code = e.response.json().get("code", 0)
                except Exception:
                    code = 0
                if code == -1021 and attempt < 2:
                    log.warning("Timestamp desincronizado, resinconizando reloj...")
                    self._sync_time()
                    if signed:
                        params = self._sign({k: v for k, v in params.items()
                                             if k not in ("timestamp", "signature")})
                    time.sleep(0.5)
                elif attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    def _post(self, path: str, params: dict = None):
        params = params or {}
        for attempt in range(3):
            try:
                signed = self._sign(dict(params))
                r      = self.session.post(f"{BASE_URL}{path}", params=signed, timeout=15)
                data   = r.json()
                if isinstance(data, dict) and data.get("code", 0) == -1021:
                    # Timestamp desincronizado → resincronizar y reintentar
                    log.warning("Timestamp desincronizado en POST, resinconizando...")
                    self._sync_time()
                    if attempt < 2:
                        continue
                if isinstance(data, dict) and data.get("code", 0) < 0:
                    raise Exception(f"Binance error {data['code']}: {data['msg']}")
                return data
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 ** attempt)
                else:
                    raise

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: str = "5m",
                    limit: int = 200, end_time: int = None) -> list:
        sym    = symbol.replace("/", "").replace(":USDT", "")
        params = {"symbol": sym, "interval": timeframe, "limit": min(limit, 1500)}
        if end_time is not None:
            params["endTime"] = end_time
        data = self._get("/fapi/v1/klines", params)
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
                    "symbol":     pos["symbol"],
                    "contracts":  abs(amt),
                    "side":       "long" if amt > 0 else "short",
                    "entryPrice": float(pos["entryPrice"]),
                })
        return result

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        sym  = symbol.replace("/", "").replace(":USDT", "")
        body = {"symbol": sym, "side": side.upper(),
                "type": "MARKET", "quantity": round(amount, 3)}
        data = self._post("/fapi/v1/order", body)
        return {"id":      str(data.get("orderId", "")),
                "average": float(data.get("avgPrice", 0) or 0)}
