from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date as date_type, datetime
from typing import Dict, List, Mapping, MutableMapping, Optional


def _normalize_date(value: Optional[object]) -> Optional[datetime]:
    """Convert strings or date objects into datetime for consistent storage."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date_type):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - invalid strings raise
            raise ValueError(f"Unsupported date string format: {value}") from exc
    raise TypeError(f"Unsupported date type: {type(value)!r}")


@dataclass(frozen=True)
class CashEvent:
    date: Optional[datetime]
    currency: str
    amount: float
    balance_after: float
    reason: str
    note: Optional[str] = None


@dataclass(frozen=True)
class Trade:
    date: datetime
    action: str
    ticker: str
    price: float
    quantity: float
    currency: str
    cost_pct: float
    gross_amount: float
    total: float
    note: Optional[str] = None
    portfolio_value: Optional[float] = None


@dataclass
class Position:
    ticker: str
    currency: str
    quantity: float = 0.0
    avg_price: float = 0.0
    last_price: Optional[float] = None

    def apply_buy(self, price: float, quantity: float) -> None:
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive for buy operations")
        total_cost = self.avg_price * self.quantity + price * quantity
        self.quantity += quantity
        self.avg_price = total_cost / self.quantity
        self.last_price = price

    def apply_sell(self, price: float, quantity: float) -> None:
        if price <= 0 or quantity <= 0:
            raise ValueError("Price and quantity must be positive for sell operations")
        if quantity > self.quantity:
            raise ValueError("Cannot sell more shares than currently held")
        self.quantity -= quantity
        self.last_price = price
        if self.quantity == 0:
            self.avg_price = 0.0

    def valuation(self, current_price: Optional[float] = None) -> Dict[str, Optional[float]]:
        effective_price: Optional[float] = current_price if current_price is not None else self.last_price
        market_value = self.quantity * effective_price if effective_price is not None else None
        unrealized = None
        if market_value is not None:
            unrealized = market_value - (self.avg_price * self.quantity)
        return {
            "ticker": self.ticker,
            "currency": self.currency,
            "quantity": self.quantity,
            "avg_price": self.avg_price,
            "last_price": effective_price,
            "market_value": market_value,
            "unrealized_pnl": unrealized,
        }


class Portfolio:
    """Simple portfolio that manages cash, stock positions, and trade history."""

    def __init__(self, initial_cash: Optional[Mapping[str, float]] = None) -> None:
        self.positions: Dict[str, Position] = {}
        self.cash_balances: MutableMapping[str, float] = defaultdict(float)
        self.initial_cash: Dict[str, float] = {}
        self.cash_events: List[CashEvent] = []
        self.trade_history: List[Trade] = []
        if initial_cash:
            for currency, amount in initial_cash.items():
                self._change_cash(
                    amount=float(amount),
                    currency=self._normalize_currency(currency),
                    reason="INITIAL",
                    date=None,
                    note="initial capital",
                )
                self.initial_cash[self._normalize_currency(currency)] = float(amount)

    @staticmethod
    def _normalize_currency(currency: str) -> str:
        if not currency:
            raise ValueError("Currency code must be provided")
        return currency.upper()

    def _change_cash(
        self,
        amount: float,
        currency: str,
        reason: str,
        date: Optional[object] = None,
        note: Optional[str] = None,
    ) -> None:
        normalized_currency = self._normalize_currency(currency)
        self.cash_balances[normalized_currency] += amount
        if self.cash_balances[normalized_currency] < -1e-9:
            raise RuntimeError(f"Cash balance for {normalized_currency} cannot be negative")
        self.cash_events.append(
            CashEvent(
                date=_normalize_date(date),
                currency=normalized_currency,
                amount=amount,
                balance_after=self.cash_balances[normalized_currency],
                reason=reason,
                note=note,
            )
        )

    def deposit(self, amount: float, currency: str, date: Optional[object] = None, note: Optional[str] = None) -> None:
        if amount <= 0:
            raise ValueError("Deposit amount must be positive")
        self._change_cash(amount, currency, reason="DEPOSIT", date=date, note=note)

    def withdraw(self, amount: float, currency: str, date: Optional[object] = None, note: Optional[str] = None) -> None:
        if amount <= 0:
            raise ValueError("Withdrawal amount must be positive")
        normalized_currency = self._normalize_currency(currency)
        if self.cash_balances[normalized_currency] + 1e-9 < amount:
            raise RuntimeError(f"Insufficient cash in {normalized_currency}")
        self._change_cash(-amount, normalized_currency, reason="WITHDRAW", date=date, note=note)

    def add_stock(self, ticker: str, currency: str) -> None:
        if ticker in self.positions:
            raise ValueError(f"Stock {ticker} already exists in the portfolio")
        self.positions[ticker] = Position(ticker=ticker, currency=self._normalize_currency(currency))

    def remove_stock(self, ticker: str) -> None:
        position = self.positions.get(ticker)
        if position is None:
            raise ValueError(f"Stock {ticker} not found in the portfolio")
        if position.quantity != 0:
            raise RuntimeError("Cannot remove a stock position that still has holdings")
        del self.positions[ticker]

    def _require_position(self, ticker: str) -> Position:
        if ticker not in self.positions:
            raise ValueError(f"Stock {ticker} does not exist. Add it before trading.")
        return self.positions[ticker]

    def buy_stock(
        self,
        ticker: str,
        date: object,
        price: float,
        quantity: float,
        currency: Optional[str] = None,
        cost_pct: float = 0.0,
        note: Optional[str] = None,
    ) -> Trade:
        if quantity <= 0 or price <= 0:
            raise ValueError("Price and quantity must be positive for buy operations")
        position = self.positions.get(ticker)
        if position is None:
            if not currency:
                raise ValueError("Currency required when buying a new stock")
            self.add_stock(ticker, currency)
            position = self.positions[ticker]
        if currency and self._normalize_currency(currency) != position.currency:
            raise ValueError("Currency mismatch for the stock position")
        total_before_costs = price * quantity
        total_cost = total_before_costs * (1 + cost_pct / 100.0)
        normalized_date = _normalize_date(date)
        self._ensure_cash(position.currency, total_cost)
        self._change_cash(-total_cost, position.currency, reason="BUY", date=normalized_date, note=note or ticker)
        position.apply_buy(price, quantity)
        trade = Trade(
            date=normalized_date or datetime.utcnow(),
            action="BUY",
            ticker=ticker,
            price=price,
            quantity=quantity,
            currency=position.currency,
            cost_pct=cost_pct,
            gross_amount=total_before_costs,
            total=-total_cost,
            note=note,
        )
        self.trade_history.append(trade)
        return trade

    def sell_stock(
        self,
        ticker: str,
        date: object,
        price: float,
        quantity: float,
        cost_pct: float = 0.0,
        note: Optional[str] = None,
    ) -> Trade:
        if quantity <= 0 or price <= 0:
            raise ValueError("Price and quantity must be positive for sell operations")
        position = self._require_position(ticker)
        if quantity > position.quantity:
            raise RuntimeError("Cannot sell more shares than currently held")
        total_before_costs = price * quantity
        total_proceeds = total_before_costs * (1 - cost_pct / 100.0)
        normalized_date = _normalize_date(date)
        position.apply_sell(price, quantity)
        self._change_cash(total_proceeds, position.currency, reason="SELL", date=normalized_date, note=note or ticker)
        trade = Trade(
            date=normalized_date or datetime.utcnow(),
            action="SELL",
            ticker=ticker,
            price=price,
            quantity=quantity,
            currency=position.currency,
            cost_pct=cost_pct,
            gross_amount=total_before_costs,
            total=total_proceeds,
            note=note,
        )
        self.trade_history.append(trade)
        return trade

    def _ensure_cash(self, currency: str, required_amount: float) -> None:
        if self.cash_balances[self._normalize_currency(currency)] + 1e-9 < required_amount:
            raise RuntimeError(f"Insufficient cash in {currency}")

    def update_prices(self, prices: Mapping[str, float]) -> None:
        for ticker, price in prices.items():
            if ticker in self.positions and price > 0:
                self.positions[ticker].last_price = price

    def get_stock_balances(self, prices: Optional[Mapping[str, float]] = None) -> Dict[str, Dict[str, Optional[float]]]:
        if prices:
            self.update_prices(prices)
        return {ticker: position.valuation() for ticker, position in self.positions.items()}

    def get_cash_balances(self) -> Dict[str, float]:
        return dict(self.cash_balances)

    def snapshot(
        self,
        prices: Optional[Mapping[str, float]] = None,
    ) -> Dict[str, object]:
        """Return a snapshot of holdings, aggregated by currency."""
        stock_balances = self.get_stock_balances(prices)
        stock_totals: MutableMapping[str, float] = defaultdict(float)
        for info in stock_balances.values():
            value = info["market_value"]
            currency = info["currency"]
            if value is not None:
                stock_totals[currency] += value
        totals_by_currency: MutableMapping[str, float] = defaultdict(float)
        for currency, amount in self.cash_balances.items():
            totals_by_currency[currency] += amount
        for currency, amount in stock_totals.items():
            totals_by_currency[currency] += amount
        return {
            "cash": self.get_cash_balances(),
            "stocks": stock_balances,
            "totals": dict(totals_by_currency),
        }

    def get_trade_history(self) -> List[Trade]:
        return list(self.trade_history)

    def get_cash_history(self) -> List[CashEvent]:
        return list(self.cash_events)

    def annotate_trades_with_value(self, target_date: datetime, portfolio_value: float) -> None:
        """Record the portfolio value on trades that happened at target_date."""
        target_day = target_date.date()
        for idx, trade in enumerate(self.trade_history):
            if trade.date.date() != target_day:
                continue
            if getattr(trade, "portfolio_value", None) == portfolio_value:
                continue
            self.trade_history[idx] = replace(trade, portfolio_value=portfolio_value)
