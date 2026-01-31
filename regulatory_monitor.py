"""
Regulatory Monitoring and Compliance System
Enterprise-grade compliance with:
- Trading limits and circuit breakers
- Position monitoring and reporting
- Suspicious activity detection
- Audit trail generation
- Regulatory reporting
- Risk limit enforcement
"""

import asyncio
import json
import time
import logging
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
from datetime import datetime, timedelta
import hashlib

logger = logging.getLogger("PolyMangoBot.regulatory")


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    HIGH = "high"
    CRITICAL = "critical"


class ComplianceStatus(Enum):
    """Compliance check status"""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    PENDING = "pending"


class ViolationType(Enum):
    """Types of compliance violations"""
    POSITION_LIMIT = "position_limit"
    DAILY_LOSS_LIMIT = "daily_loss_limit"
    TRADE_FREQUENCY = "trade_frequency"
    ORDER_SIZE = "order_size"
    WASH_TRADING = "wash_trading"
    MARKET_MANIPULATION = "market_manipulation"
    CONCENTRATION = "concentration"
    LEVERAGE = "leverage"
    UNAUTHORIZED_VENUE = "unauthorized_venue"


@dataclass
class ComplianceRule:
    """A compliance rule definition"""
    rule_id: str
    name: str
    description: str
    threshold: float
    violation_type: ViolationType
    severity: AlertSeverity
    enabled: bool = True
    cooldown_minutes: int = 5

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "enabled": self.enabled
        }


@dataclass
class ComplianceAlert:
    """A compliance alert"""
    alert_id: str
    rule_id: str
    violation_type: ViolationType
    severity: AlertSeverity
    message: str
    details: Dict
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False
    resolution_notes: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "rule_id": self.rule_id,
            "violation_type": self.violation_type.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolution_notes": self.resolution_notes
        }


@dataclass
class AuditRecord:
    """Audit trail record"""
    record_id: str
    action_type: str
    actor: str
    target: str
    details: Dict
    timestamp: float
    ip_address: Optional[str] = None
    session_id: Optional[str] = None
    checksum: str = ""

    def __post_init__(self):
        if not self.checksum:
            self.checksum = self._calculate_checksum()

    def _calculate_checksum(self) -> str:
        """Calculate integrity checksum"""
        data = f"{self.record_id}{self.action_type}{self.actor}{self.target}{self.timestamp}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def verify_integrity(self) -> bool:
        """Verify record integrity"""
        return self.checksum == self._calculate_checksum()

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "action_type": self.action_type,
            "actor": self.actor,
            "target": self.target,
            "details": self.details,
            "timestamp": self.timestamp,
            "ip_address": self.ip_address,
            "session_id": self.session_id,
            "checksum": self.checksum
        }


@dataclass
class PositionReport:
    """Position status report"""
    market: str
    venue: str
    position_size: float
    position_value: float
    unrealized_pnl: float
    realized_pnl: float
    daily_volume: float
    max_position_today: float
    timestamp: float

    def to_dict(self) -> Dict:
        return {
            "market": self.market,
            "venue": self.venue,
            "position_size": self.position_size,
            "position_value": self.position_value,
            "unrealized_pnl": self.unrealized_pnl,
            "realized_pnl": self.realized_pnl,
            "daily_volume": self.daily_volume,
            "max_position_today": self.max_position_today,
            "timestamp": self.timestamp
        }


class TradingLimits:
    """Manages trading limits"""

    def __init__(self):
        # Position limits
        self.max_position_per_market = 10000  # USD
        self.max_total_position = 50000  # USD
        self.max_position_concentration = 0.3  # 30% of total

        # Loss limits
        self.max_daily_loss = 1000  # USD
        self.max_weekly_loss = 3000  # USD
        self.max_monthly_loss = 10000  # USD

        # Trade limits
        self.max_order_size = 5000  # USD
        self.max_trades_per_minute = 10
        self.max_trades_per_hour = 100
        self.max_trades_per_day = 500

        # Leverage limits
        self.max_leverage = 3.0

        # Approved venues
        self.approved_venues: Set[str] = {
            "polymarket", "kraken", "coinbase"
        }

    def to_dict(self) -> Dict:
        return {
            "max_position_per_market": self.max_position_per_market,
            "max_total_position": self.max_total_position,
            "max_position_concentration": self.max_position_concentration,
            "max_daily_loss": self.max_daily_loss,
            "max_weekly_loss": self.max_weekly_loss,
            "max_monthly_loss": self.max_monthly_loss,
            "max_order_size": self.max_order_size,
            "max_trades_per_minute": self.max_trades_per_minute,
            "max_trades_per_hour": self.max_trades_per_hour,
            "max_trades_per_day": self.max_trades_per_day,
            "max_leverage": self.max_leverage,
            "approved_venues": list(self.approved_venues)
        }


class PositionTracker:
    """Tracks positions and P&L"""

    def __init__(self):
        self._positions: Dict[str, Dict] = {}  # market -> position
        self._daily_pnl: float = 0.0
        self._weekly_pnl: float = 0.0
        self._monthly_pnl: float = 0.0
        self._trade_history: deque = deque(maxlen=10000)
        self._pnl_history: deque = deque(maxlen=10000)

        # Tracking periods
        self._day_start = self._get_day_start()
        self._week_start = self._get_week_start()
        self._month_start = self._get_month_start()

    def _get_day_start(self) -> float:
        """Get start of current day"""
        now = datetime.now()
        return datetime(now.year, now.month, now.day).timestamp()

    def _get_week_start(self) -> float:
        """Get start of current week"""
        now = datetime.now()
        week_start = now - timedelta(days=now.weekday())
        return datetime(week_start.year, week_start.month, week_start.day).timestamp()

    def _get_month_start(self) -> float:
        """Get start of current month"""
        now = datetime.now()
        return datetime(now.year, now.month, 1).timestamp()

    def update_position(self, market: str, venue: str, size: float, price: float):
        """Update position for a market"""
        key = f"{market}_{venue}"

        if key not in self._positions:
            self._positions[key] = {
                "market": market,
                "venue": venue,
                "size": 0,
                "avg_price": 0,
                "value": 0,
                "unrealized_pnl": 0,
                "max_size_today": 0
            }

        pos = self._positions[key]

        # Update average price
        if pos["size"] + size != 0:
            if (pos["size"] >= 0 and size > 0) or (pos["size"] <= 0 and size < 0):
                # Adding to position
                total_cost = pos["size"] * pos["avg_price"] + size * price
                new_size = pos["size"] + size
                pos["avg_price"] = total_cost / new_size if new_size != 0 else price
            else:
                # Reducing position - realize P&L
                reduced_size = min(abs(size), abs(pos["size"]))
                pnl = reduced_size * (price - pos["avg_price"]) * (1 if pos["size"] > 0 else -1)
                self._record_pnl(pnl)

        pos["size"] += size
        pos["value"] = abs(pos["size"]) * price
        pos["max_size_today"] = max(pos["max_size_today"], abs(pos["size"]))

    def record_trade(self, trade: Dict):
        """Record a completed trade"""
        self._trade_history.append({
            **trade,
            "timestamp": time.time()
        })

        # Check for period reset
        self._check_period_reset()

    def _record_pnl(self, pnl: float):
        """Record realized P&L"""
        self._daily_pnl += pnl
        self._weekly_pnl += pnl
        self._monthly_pnl += pnl

        self._pnl_history.append({
            "pnl": pnl,
            "cumulative_daily": self._daily_pnl,
            "timestamp": time.time()
        })

    def _check_period_reset(self):
        """Check if P&L periods need reset"""
        now = time.time()

        if now > self._day_start + 86400:
            self._daily_pnl = 0
            self._day_start = self._get_day_start()
            # Reset max position today
            for pos in self._positions.values():
                pos["max_size_today"] = abs(pos["size"])

        if now > self._week_start + 604800:
            self._weekly_pnl = 0
            self._week_start = self._get_week_start()

        if now > self._month_start + 2592000:
            self._monthly_pnl = 0
            self._month_start = self._get_month_start()

    def get_position(self, market: str, venue: str) -> Optional[Dict]:
        """Get position for a market"""
        key = f"{market}_{venue}"
        return self._positions.get(key)

    def get_total_position_value(self) -> float:
        """Get total position value across all markets"""
        return sum(pos["value"] for pos in self._positions.values())

    def get_daily_pnl(self) -> float:
        """Get current day P&L"""
        self._check_period_reset()
        return self._daily_pnl

    def get_weekly_pnl(self) -> float:
        """Get current week P&L"""
        self._check_period_reset()
        return self._weekly_pnl

    def get_monthly_pnl(self) -> float:
        """Get current month P&L"""
        self._check_period_reset()
        return self._monthly_pnl

    def get_trade_count(self, minutes: int = 1) -> int:
        """Get trade count in last N minutes"""
        cutoff = time.time() - minutes * 60
        return sum(1 for t in self._trade_history if t["timestamp"] > cutoff)

    def get_position_report(self) -> List[PositionReport]:
        """Generate position reports"""
        reports = []
        for key, pos in self._positions.items():
            reports.append(PositionReport(
                market=pos["market"],
                venue=pos["venue"],
                position_size=pos["size"],
                position_value=pos["value"],
                unrealized_pnl=pos["unrealized_pnl"],
                realized_pnl=self._daily_pnl,  # Simplified
                daily_volume=self._get_daily_volume(pos["market"], pos["venue"]),
                max_position_today=pos["max_size_today"],
                timestamp=time.time()
            ))
        return reports

    def _get_daily_volume(self, market: str, venue: str) -> float:
        """Get daily trading volume for a market"""
        cutoff = self._day_start
        volume = 0
        for trade in self._trade_history:
            if (trade.get("market") == market and
                trade.get("venue") == venue and
                trade.get("timestamp", 0) > cutoff):
                volume += abs(trade.get("quantity", 0) * trade.get("price", 0))
        return volume


class WashTradeDetector:
    """Detects potential wash trading"""

    def __init__(self, time_window_seconds: int = 60, price_tolerance: float = 0.001):
        self.time_window = time_window_seconds
        self.price_tolerance = price_tolerance
        self._recent_trades: deque = deque(maxlen=1000)

    def check_trade(self, trade: Dict) -> Tuple[bool, float]:
        """
        Check if trade is potential wash trade.
        Returns (is_wash_trade, confidence)
        """
        self._recent_trades.append({
            **trade,
            "timestamp": time.time()
        })

        # Look for matching opposite trades
        market = trade.get("market")
        side = trade.get("side")
        price = trade.get("price", 0)
        quantity = trade.get("quantity", 0)
        timestamp = trade.get("timestamp", time.time())

        opposite_side = "sell" if side == "buy" else "buy"
        cutoff = timestamp - self.time_window

        suspicious_matches = []

        for t in self._recent_trades:
            if t.get("market") != market:
                continue
            if t.get("side") != opposite_side:
                continue
            if t.get("timestamp", 0) < cutoff:
                continue

            # Check price similarity
            t_price = t.get("price", 0)
            if t_price == 0:
                continue

            price_diff = abs(price - t_price) / t_price
            if price_diff > self.price_tolerance:
                continue

            # Check quantity similarity
            t_qty = t.get("quantity", 0)
            if t_qty == 0:
                continue

            qty_diff = abs(quantity - t_qty) / max(quantity, t_qty)
            if qty_diff < 0.2:  # Within 20%
                suspicious_matches.append({
                    "trade": t,
                    "price_diff": price_diff,
                    "qty_diff": qty_diff
                })

        if suspicious_matches:
            # Calculate confidence based on similarity
            best_match = min(suspicious_matches, key=lambda x: x["price_diff"] + x["qty_diff"])
            confidence = 1 - (best_match["price_diff"] + best_match["qty_diff"]) / 2
            return True, confidence

        return False, 0.0


class AuditTrail:
    """Maintains audit trail for compliance"""

    def __init__(self, max_records: int = 100000):
        self._records: deque = deque(maxlen=max_records)
        self._record_counter = 0

    def log(
        self,
        action_type: str,
        actor: str,
        target: str,
        details: Dict,
        ip_address: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> AuditRecord:
        """Log an auditable action"""
        self._record_counter += 1
        record_id = f"AUD{self._record_counter:010d}"

        record = AuditRecord(
            record_id=record_id,
            action_type=action_type,
            actor=actor,
            target=target,
            details=details,
            timestamp=time.time(),
            ip_address=ip_address,
            session_id=session_id
        )

        self._records.append(record)
        logger.info(f"Audit: [{action_type}] {actor} -> {target}")

        return record

    def get_records(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        action_type: Optional[str] = None,
        actor: Optional[str] = None,
        limit: int = 100
    ) -> List[AuditRecord]:
        """Query audit records"""
        results = []

        for record in reversed(self._records):
            if len(results) >= limit:
                break

            if start_time and record.timestamp < start_time:
                continue
            if end_time and record.timestamp > end_time:
                continue
            if action_type and record.action_type != action_type:
                continue
            if actor and record.actor != actor:
                continue

            results.append(record)

        return results

    def export_records(self, filepath: str, start_time: float, end_time: float):
        """Export records to JSON file"""
        records = self.get_records(
            start_time=start_time,
            end_time=end_time,
            limit=100000
        )

        with open(filepath, 'w') as f:
            json.dump([r.to_dict() for r in records], f, indent=2)

        logger.info(f"Exported {len(records)} audit records to {filepath}")

    def verify_integrity(self) -> Dict:
        """Verify integrity of all records"""
        total = len(self._records)
        valid = sum(1 for r in self._records if r.verify_integrity())

        return {
            "total_records": total,
            "valid_records": valid,
            "integrity_rate": valid / total * 100 if total > 0 else 100,
            "timestamp": time.time()
        }


class RegulatoryMonitor:
    """
    Comprehensive regulatory monitoring and compliance system.

    Features:
    - Trading limit enforcement
    - Position monitoring
    - Wash trade detection
    - Audit trail
    - Compliance reporting
    - Alert management
    """

    def __init__(self):
        self.limits = TradingLimits()
        self.position_tracker = PositionTracker()
        self.wash_detector = WashTradeDetector()
        self.audit_trail = AuditTrail()

        # Compliance rules
        self._rules: Dict[str, ComplianceRule] = {}
        self._init_default_rules()

        # Alerts
        self._alerts: List[ComplianceAlert] = []
        self._alert_counter = 0

        # Circuit breaker
        self._trading_halted = False
        self._halt_reason: Optional[str] = None

    def _init_default_rules(self):
        """Initialize default compliance rules"""
        rules = [
            ComplianceRule(
                rule_id="POS001",
                name="Position Limit",
                description="Maximum position per market",
                threshold=self.limits.max_position_per_market,
                violation_type=ViolationType.POSITION_LIMIT,
                severity=AlertSeverity.HIGH
            ),
            ComplianceRule(
                rule_id="POS002",
                name="Total Position Limit",
                description="Maximum total position across all markets",
                threshold=self.limits.max_total_position,
                violation_type=ViolationType.POSITION_LIMIT,
                severity=AlertSeverity.CRITICAL
            ),
            ComplianceRule(
                rule_id="LOSS001",
                name="Daily Loss Limit",
                description="Maximum daily loss",
                threshold=self.limits.max_daily_loss,
                violation_type=ViolationType.DAILY_LOSS_LIMIT,
                severity=AlertSeverity.CRITICAL
            ),
            ComplianceRule(
                rule_id="FREQ001",
                name="Trade Frequency - Minute",
                description="Maximum trades per minute",
                threshold=self.limits.max_trades_per_minute,
                violation_type=ViolationType.TRADE_FREQUENCY,
                severity=AlertSeverity.WARNING
            ),
            ComplianceRule(
                rule_id="SIZE001",
                name="Order Size Limit",
                description="Maximum order size",
                threshold=self.limits.max_order_size,
                violation_type=ViolationType.ORDER_SIZE,
                severity=AlertSeverity.HIGH
            ),
            ComplianceRule(
                rule_id="WASH001",
                name="Wash Trading Detection",
                description="Potential wash trade detected",
                threshold=0.8,  # Confidence threshold
                violation_type=ViolationType.WASH_TRADING,
                severity=AlertSeverity.CRITICAL
            ),
        ]

        for rule in rules:
            self._rules[rule.rule_id] = rule

    async def check_pre_trade(self, order: Dict) -> Tuple[bool, List[str]]:
        """
        Pre-trade compliance check.
        Returns (is_allowed, list of violations)
        """
        violations = []

        # Check if trading is halted
        if self._trading_halted:
            violations.append(f"Trading halted: {self._halt_reason}")
            return False, violations

        # Check venue
        venue = order.get("venue", "").lower()
        if venue not in self.limits.approved_venues:
            violations.append(f"Unauthorized venue: {venue}")
            self._create_alert(
                "VEN001",
                ViolationType.UNAUTHORIZED_VENUE,
                AlertSeverity.CRITICAL,
                f"Trade attempted on unauthorized venue: {venue}",
                {"order": order}
            )

        # Check order size
        order_value = order.get("quantity", 0) * order.get("price", 0)
        if order_value > self.limits.max_order_size:
            violations.append(f"Order size ${order_value:.2f} exceeds limit ${self.limits.max_order_size}")

        # Check position limit
        market = order.get("market", "")
        current_pos = self.position_tracker.get_position(market, venue)
        if current_pos:
            new_value = current_pos["value"] + order_value
            if new_value > self.limits.max_position_per_market:
                violations.append(f"Position would exceed market limit")

        # Check total position
        total_pos = self.position_tracker.get_total_position_value()
        if total_pos + order_value > self.limits.max_total_position:
            violations.append(f"Total position would exceed limit")

        # Check trade frequency
        trades_per_minute = self.position_tracker.get_trade_count(minutes=1)
        if trades_per_minute >= self.limits.max_trades_per_minute:
            violations.append(f"Trade frequency limit exceeded ({trades_per_minute}/min)")

        # Check daily loss
        daily_pnl = self.position_tracker.get_daily_pnl()
        if daily_pnl < -self.limits.max_daily_loss:
            violations.append(f"Daily loss limit exceeded (${-daily_pnl:.2f})")
            self._halt_trading("Daily loss limit exceeded")

        # Log audit
        self.audit_trail.log(
            action_type="PRE_TRADE_CHECK",
            actor="system",
            target=f"{market}_{venue}",
            details={
                "order": order,
                "passed": len(violations) == 0,
                "violations": violations
            }
        )

        return len(violations) == 0, violations

    async def check_post_trade(self, trade: Dict):
        """Post-trade compliance check and recording"""
        # Record trade
        self.position_tracker.record_trade(trade)

        # Update position
        market = trade.get("market", "")
        venue = trade.get("venue", "")
        side = trade.get("side", "")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)

        size_delta = quantity if side == "buy" else -quantity
        self.position_tracker.update_position(market, venue, size_delta, price)

        # Check for wash trading
        is_wash, confidence = self.wash_detector.check_trade(trade)
        if is_wash and confidence > 0.8:
            self._create_alert(
                "WASH001",
                ViolationType.WASH_TRADING,
                AlertSeverity.CRITICAL,
                f"Potential wash trade detected (confidence: {confidence:.0%})",
                {"trade": trade, "confidence": confidence}
            )

        # Log audit
        self.audit_trail.log(
            action_type="TRADE_EXECUTED",
            actor="bot",
            target=f"{market}_{venue}",
            details={
                "trade": trade,
                "position_after": self.position_tracker.get_position(market, venue)
            }
        )

    def _create_alert(
        self,
        rule_id: str,
        violation_type: ViolationType,
        severity: AlertSeverity,
        message: str,
        details: Dict
    ):
        """Create a compliance alert"""
        self._alert_counter += 1
        alert_id = f"ALT{self._alert_counter:08d}"

        alert = ComplianceAlert(
            alert_id=alert_id,
            rule_id=rule_id,
            violation_type=violation_type,
            severity=severity,
            message=message,
            details=details,
            timestamp=time.time()
        )

        self._alerts.append(alert)

        # Log based on severity
        if severity == AlertSeverity.CRITICAL:
            logger.critical(f"COMPLIANCE ALERT: {message}")
        elif severity == AlertSeverity.HIGH:
            logger.error(f"COMPLIANCE ALERT: {message}")
        elif severity == AlertSeverity.WARNING:
            logger.warning(f"COMPLIANCE ALERT: {message}")
        else:
            logger.info(f"COMPLIANCE ALERT: {message}")

        # Audit
        self.audit_trail.log(
            action_type="ALERT_CREATED",
            actor="system",
            target=rule_id,
            details=alert.to_dict()
        )

    def _halt_trading(self, reason: str):
        """Halt all trading"""
        self._trading_halted = True
        self._halt_reason = reason

        logger.critical(f"TRADING HALTED: {reason}")

        self.audit_trail.log(
            action_type="TRADING_HALTED",
            actor="system",
            target="all",
            details={"reason": reason}
        )

    def resume_trading(self, authorized_by: str, notes: str = ""):
        """Resume trading after halt"""
        if not self._trading_halted:
            return

        self._trading_halted = False
        reason = self._halt_reason
        self._halt_reason = None

        logger.info(f"Trading resumed by {authorized_by}")

        self.audit_trail.log(
            action_type="TRADING_RESUMED",
            actor=authorized_by,
            target="all",
            details={
                "previous_halt_reason": reason,
                "notes": notes
            }
        )

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str):
        """Acknowledge an alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True

                self.audit_trail.log(
                    action_type="ALERT_ACKNOWLEDGED",
                    actor=acknowledged_by,
                    target=alert_id,
                    details={"alert": alert.to_dict()}
                )
                return True
        return False

    def resolve_alert(self, alert_id: str, resolved_by: str, notes: str = ""):
        """Resolve an alert"""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.resolved = True
                alert.resolution_notes = notes

                self.audit_trail.log(
                    action_type="ALERT_RESOLVED",
                    actor=resolved_by,
                    target=alert_id,
                    details={
                        "alert": alert.to_dict(),
                        "notes": notes
                    }
                )
                return True
        return False

    def get_compliance_status(self) -> Dict:
        """Get overall compliance status"""
        daily_pnl = self.position_tracker.get_daily_pnl()
        total_position = self.position_tracker.get_total_position_value()

        # Check all limits
        checks = {
            "daily_loss": {
                "status": ComplianceStatus.PASSED if daily_pnl > -self.limits.max_daily_loss else ComplianceStatus.FAILED,
                "current": -daily_pnl,
                "limit": self.limits.max_daily_loss
            },
            "total_position": {
                "status": ComplianceStatus.PASSED if total_position < self.limits.max_total_position else ComplianceStatus.FAILED,
                "current": total_position,
                "limit": self.limits.max_total_position
            },
            "trading_status": {
                "status": ComplianceStatus.PASSED if not self._trading_halted else ComplianceStatus.FAILED,
                "halted": self._trading_halted,
                "halt_reason": self._halt_reason
            }
        }

        # Count open alerts
        open_alerts = {
            "critical": sum(1 for a in self._alerts if a.severity == AlertSeverity.CRITICAL and not a.resolved),
            "high": sum(1 for a in self._alerts if a.severity == AlertSeverity.HIGH and not a.resolved),
            "warning": sum(1 for a in self._alerts if a.severity == AlertSeverity.WARNING and not a.resolved),
            "info": sum(1 for a in self._alerts if a.severity == AlertSeverity.INFO and not a.resolved),
        }

        # Overall status
        overall = ComplianceStatus.PASSED
        if any(c.get("status") == ComplianceStatus.FAILED for c in checks.values()):
            overall = ComplianceStatus.FAILED
        elif open_alerts["critical"] > 0 or open_alerts["high"] > 0:
            overall = ComplianceStatus.WARNING

        return {
            "overall_status": overall.value,
            "checks": {k: {"status": v["status"].value if "status" in v else None, **{kk: vv for kk, vv in v.items() if kk != "status"}} for k, v in checks.items()},
            "open_alerts": open_alerts,
            "trading_halted": self._trading_halted,
            "timestamp": time.time()
        }

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get compliance alerts"""
        alerts = []

        for alert in reversed(self._alerts):
            if len(alerts) >= limit:
                break
            if severity and alert.severity != severity:
                continue
            if resolved is not None and alert.resolved != resolved:
                continue
            alerts.append(alert.to_dict())

        return alerts

    def get_position_reports(self) -> List[Dict]:
        """Get position reports"""
        return [r.to_dict() for r in self.position_tracker.get_position_report()]

    def generate_daily_report(self) -> Dict:
        """Generate daily compliance report"""
        now = datetime.now()
        report_date = now.strftime("%Y-%m-%d")

        # Get 24-hour stats
        day_start = (now - timedelta(days=1)).timestamp()

        alerts_24h = [a for a in self._alerts if a.timestamp > day_start]
        audit_records = self.audit_trail.get_records(
            start_time=day_start,
            limit=10000
        )

        report = {
            "report_date": report_date,
            "generated_at": time.time(),
            "compliance_status": self.get_compliance_status(),
            "positions": self.get_position_reports(),
            "pnl_summary": {
                "daily": self.position_tracker.get_daily_pnl(),
                "weekly": self.position_tracker.get_weekly_pnl(),
                "monthly": self.position_tracker.get_monthly_pnl()
            },
            "alerts_24h": {
                "total": len(alerts_24h),
                "critical": sum(1 for a in alerts_24h if a.severity == AlertSeverity.CRITICAL),
                "high": sum(1 for a in alerts_24h if a.severity == AlertSeverity.HIGH),
                "resolved": sum(1 for a in alerts_24h if a.resolved)
            },
            "audit_summary": {
                "total_records": len(audit_records),
                "integrity_check": self.audit_trail.verify_integrity()
            },
            "limits": self.limits.to_dict()
        }

        # Log report generation
        self.audit_trail.log(
            action_type="DAILY_REPORT_GENERATED",
            actor="system",
            target="compliance",
            details={"report_date": report_date}
        )

        return report


# Test function
async def test_regulatory_monitor():
    """Test the regulatory monitor"""
    monitor = RegulatoryMonitor()

    print("Testing Regulatory Monitor...\n")

    # Test pre-trade checks
    test_orders = [
        {
            "market": "BTC",
            "venue": "kraken",
            "side": "buy",
            "quantity": 0.5,
            "price": 40000
        },
        {
            "market": "BTC",
            "venue": "unauthorized_exchange",  # Should fail
            "side": "buy",
            "quantity": 0.1,
            "price": 40000
        },
        {
            "market": "ETH",
            "venue": "coinbase",
            "side": "buy",
            "quantity": 100,
            "price": 100  # Should fail - exceeds order size
        }
    ]

    print("Pre-trade Checks:")
    for order in test_orders:
        allowed, violations = await monitor.check_pre_trade(order)
        print(f"  Order: {order['market']} on {order['venue']}")
        print(f"  Allowed: {allowed}")
        if violations:
            print(f"  Violations: {violations}")
        print()

    # Simulate some trades
    print("Simulating trades...")
    for i in range(5):
        trade = {
            "trade_id": f"trade_{i}",
            "market": "BTC",
            "venue": "kraken",
            "side": "buy" if i % 2 == 0 else "sell",
            "quantity": 0.1,
            "price": 40000 + i * 100,
            "timestamp": time.time()
        }
        await monitor.check_post_trade(trade)
        await asyncio.sleep(0.1)

    # Get status
    print("\nCompliance Status:")
    status = monitor.get_compliance_status()
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Get alerts
    print("\nAlerts:")
    alerts = monitor.get_alerts(limit=5)
    for alert in alerts:
        print(f"  [{alert['severity']}] {alert['message']}")

    # Generate daily report
    print("\nDaily Report:")
    report = monitor.generate_daily_report()
    print(f"  Date: {report['report_date']}")
    print(f"  Overall Status: {report['compliance_status']['overall_status']}")
    print(f"  Daily P&L: ${report['pnl_summary']['daily']:.2f}")
    print(f"  Alerts (24h): {report['alerts_24h']['total']}")


if __name__ == "__main__":
    asyncio.run(test_regulatory_monitor())
