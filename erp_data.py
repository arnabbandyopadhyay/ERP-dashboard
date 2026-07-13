"""
erp_data.py — Realistic ERP-scale dataset generator.
Produces: 60 suppliers, 500 purchase orders, 120 inventory SKUs, 24-month spend series.
Dependencies: numpy, stdlib only.
"""

from __future__ import annotations
import random
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import numpy as np


# ── Seeded RNG for reproducibility ───────────────────────────────
RNG = random.Random(0xDEADBEEF)
NP_RNG = np.random.default_rng(42)


# ── Constants ─────────────────────────────────────────────────────
SUPPLIER_NAMES = [
    "Siemens Industrial AG",      "Bosch Supply Chain GmbH",    "BASF Chemicals GmbH",
    "ThyssenKrupp Materials",     "Freudenberg Sealing Tech",   "ZF Friedrichshafen AG",
    "Schaeffler Technologies",    "Continental Automotive",     "Daimler Truck Parts",
    "Henkel Adhesives GmbH",     "Wacker Chemie AG",           "Lanxess Specialty Chem",
    "Evonik Industries SE",       "Covestro Polymers GmbH",    "SGL Carbon SE",
    "Voith GmbH & Co KG",        "Knorr-Bremse AG",           "Mahle Group GmbH",
    "Mann+Hummel Group",          "Leoni Wire Systems",         "Infineon Munich GmbH",
    "STMicroelectronics EU",      "NXP Semiconductors NL",     "Rohm Semiconductor EU",
    "Murata Manufacturing EU",    "TDK Europe GmbH",            "Alps Alpine GmbH",
    "TE Connectivity DE",         "Molex LLC Europe",           "Amphenol Tuchel",
    "Belden Cables GmbH",         "Phoenix Contact GmbH",       "Weidmuller Interface",
    "Rittal Enclosures GmbH",    "Lapp Group Stuttgart",       "Helukabel GmbH",
    "Igus GmbH Cologne",          "Norgren Pneumatics",         "Festo AG & Co",
    "Parker Hannifin Europe",     "SMC Corporation Europe",     "Schunk GmbH & Co",
    "Zimmer Group GmbH",          "Hiwin Technologies EU",      "Renishaw Europe Ltd",
    "Hexagon Metrology GmbH",    "Carl Zeiss Meditec",         "Kistler Group CH",
    "Heidenhain GmbH",            "Sick AG Waldkirch",          "Balluff GmbH",
    "Turck GmbH & Co",            "Pepperl+Fuchs SE",           "IFM Electronic GmbH",
    "Baumer Group CH",            "Pilz GmbH & Co KG",          "Omron Europe BV",
    "Keyence Germany GmbH",       "Cognex Europe GmbH",         "Datalogic EU SRL",
    "Endress+Hauser Group",
]

COUNTRIES = ["DE","US","JP","CN","FR","IT","KR","GB","NL","CH","SE","AT"]
CATEGORIES = ["Electronics","Mechanical Parts","Chemicals","Logistics","Packaging",
               "IT Services","MRO","Raw Materials"]
PLANTS = ["Frankfurt HQ","Munich Plant","Hamburg Port","Stuttgart R&D","Berlin Office"]
APPROVERS = ["H. Keller","M. Schreiber","K. Zimmermann","A. Fischer","T. Bauer"]
ANOMALY_TYPES = ["price_spike","duplicate_po","unusual_quantity",
                  "payment_breach","unapproved_supplier","three_way_mismatch"]
STATUSES = ["APPROVED","PENDING","DELIVERED","CANCELLED","DISPUTED"]
PO_STATUS_WEIGHTS = [0.40, 0.20, 0.25, 0.08, 0.07]


# ── Data classes ──────────────────────────────────────────────────
@dataclass
class Supplier:
    id:             str
    name:           str
    country:        str
    tier:           int           # 1=critical, 2=preferred, 3=approved
    risk_score:     int           # 0–100
    on_time_rate:   float         # 0–1
    quality_score:  float         # 0–1
    spend_ytd:      float
    lead_time_days: int
    payment_terms:  int           # Net days
    category:       str
    contracts:      int
    incidents:      int
    iso9001:        bool
    iso14001:       bool
    connections:    list[str] = field(default_factory=list)   # related supplier IDs

    def risk_label(self) -> str:
        if self.risk_score > 70: return "HIGH"
        if self.risk_score > 40: return "MEDIUM"
        return "LOW"

    def status_color(self) -> str:
        """Rich color string based on risk."""
        if self.risk_score > 70: return "red"
        if self.risk_score > 40: return "yellow"
        return "green"

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name, "country": self.country,
            "tier": self.tier, "risk_score": self.risk_score,
            "on_time_rate": round(self.on_time_rate, 3),
            "quality_score": round(self.quality_score, 3),
            "spend_ytd": self.spend_ytd,
            "lead_time_days": self.lead_time_days,
            "payment_terms": self.payment_terms,
            "category": self.category, "contracts": self.contracts,
            "incidents": self.incidents, "iso9001": self.iso9001,
            "iso14001": self.iso14001,
        }


@dataclass
class PurchaseOrder:
    id:                str
    supplier_id:       str
    supplier_name:     str
    category:          str
    plant:             str
    amount:            float
    currency:          str
    date:              str
    status:            str
    payment_days:      int
    quantity:          int
    unit_price:        float
    lead_time_actual:  int
    lead_time_planned: int
    is_anomaly:        bool
    anomaly_type:      Optional[str]
    anomaly_score:     float        # Isolation Forest score (filled later)
    invoice_match:     bool
    three_way_match:   bool
    approver:          str
    budget_code:       str

    def variance_pct(self) -> float:
        if self.lead_time_planned == 0:
            return 0.0
        return (self.lead_time_actual - self.lead_time_planned) / self.lead_time_planned * 100

    def to_feature_vector(self) -> list[float]:
        """Numeric features for Isolation Forest."""
        return [
            self.amount / 1_000_000,
            self.payment_days / 90,
            self.lead_time_actual / max(self.lead_time_planned, 1),
            self.quantity / 1000,
            float(not self.invoice_match),
            float(not self.three_way_match),
            self.unit_price / 1000,
        ]

    def to_dict(self) -> dict:
        return {
            "id": self.id, "supplier_id": self.supplier_id,
            "supplier_name": self.supplier_name, "category": self.category,
            "plant": self.plant, "amount": self.amount, "currency": self.currency,
            "date": self.date, "status": self.status,
            "payment_days": self.payment_days, "quantity": self.quantity,
            "unit_price": self.unit_price, "is_anomaly": self.is_anomaly,
            "anomaly_type": self.anomaly_type, "anomaly_score": self.anomaly_score,
            "invoice_match": self.invoice_match, "three_way_match": self.three_way_match,
            "approver": self.approver,
        }


@dataclass
class InventorySKU:
    sku:          str
    description:  str
    category:     str
    plant:        str
    qty_on_hand:  int
    reorder_pt:   int
    qty_on_order: int
    unit_cost:    float
    lead_time:    int
    abc_class:    str    # A=high value, B=medium, C=low
    last_move:    str
    supplier_id:  str

    def below_reorder(self) -> bool:
        return self.qty_on_hand < self.reorder_pt

    def coverage_days(self) -> float:
        if self.qty_on_hand == 0:
            return 0.0
        daily_usage = max(1, self.reorder_pt / 30)
        return self.qty_on_hand / daily_usage


@dataclass
class MonthlyKPI:
    month:            str   # YYYY-MM
    spend:            float
    pos:              int
    active_suppliers: int
    savings:          float
    on_time_pct:      float
    anomaly_count:    int
    disputed_value:   float


@dataclass
class ERPDataset:
    suppliers:      list[Supplier]
    purchase_orders: list[PurchaseOrder]
    inventory:      list[InventorySKU]
    monthly_kpis:   list[MonthlyKPI]

    # Quick-access indexes
    supplier_map:   dict[str, Supplier]    = field(default_factory=dict)
    po_by_supplier: dict[str, list[PurchaseOrder]] = field(default_factory=dict)

    def __post_init__(self):
        self.supplier_map = {s.id: s for s in self.suppliers}
        self.po_by_supplier = {}
        for po in self.purchase_orders:
            self.po_by_supplier.setdefault(po.supplier_id, []).append(po)

    def total_spend(self) -> float:
        return sum(po.amount for po in self.purchase_orders)

    def anomalies(self) -> list[PurchaseOrder]:
        return [po for po in self.purchase_orders if po.is_anomaly]

    def high_risk_suppliers(self, threshold: int = 70) -> list[Supplier]:
        return [s for s in self.suppliers if s.risk_score > threshold]

    def pending_pos(self) -> list[PurchaseOrder]:
        return [po for po in self.purchase_orders if po.status == "PENDING"]

    def spend_by_category(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for po in self.purchase_orders:
            result[po.category] = result.get(po.category, 0) + po.amount
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    def spend_by_plant(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for po in self.purchase_orders:
            result[po.plant] = result.get(po.plant, 0) + po.amount
        return result

    def summary(self) -> dict:
        anom = self.anomalies()
        high_risk = self.high_risk_suppliers()
        return {
            "total_spend":        self.total_spend(),
            "total_pos":          len(self.purchase_orders),
            "total_suppliers":    len(self.suppliers),
            "anomaly_count":      len(anom),
            "anomaly_rate_pct":   len(anom) / max(1, len(self.purchase_orders)) * 100,
            "high_risk_suppliers":len(high_risk),
            "pending_pos":        len(self.pending_pos()),
            "pending_value":      sum(p.amount for p in self.pending_pos()),
            "avg_on_time_pct":    sum(s.on_time_rate for s in self.suppliers) / len(self.suppliers) * 100,
            "inventory_skus":     len(self.inventory),
            "below_reorder":      sum(1 for i in self.inventory if i.below_reorder()),
            "total_savings_ytd":  sum(m.savings for m in self.monthly_kpis),
        }


# ── Generator ─────────────────────────────────────────────────────
def generate_erp_data() -> ERPDataset:
    now = datetime.now()

    # ── Suppliers ─────────────────────────────────────────────────
    suppliers: list[Supplier] = []
    for i, name in enumerate(SUPPLIER_NAMES):
        tier = 1 if RNG.random() > 0.65 else (2 if RNG.random() > 0.40 else 3)
        sup = Supplier(
            id=f"SUP-{i+1:04d}",
            name=name,
            country=RNG.choice(COUNTRIES),
            tier=tier,
            risk_score=RNG.randint(5, 98),
            on_time_rate=round(RNG.uniform(0.65, 0.99), 3),
            quality_score=round(RNG.uniform(0.70, 0.99), 3),
            spend_ytd=round(RNG.uniform(120_000, 9_500_000), 2),
            lead_time_days=RNG.randint(4, 62),
            payment_terms=RNG.choice([30, 45, 60, 90]),
            category=RNG.choice(CATEGORIES),
            contracts=RNG.randint(1, 15),
            incidents=RNG.randint(0, 7),
            iso9001=RNG.random() > 0.25,
            iso14001=RNG.random() > 0.45,
        )
        suppliers.append(sup)

    # Build supplier knowledge graph connections
    sup_ids = [s.id for s in suppliers]
    for s in suppliers:
        n_connections = RNG.randint(1, 5)
        others = [sid for sid in sup_ids if sid != s.id]
        s.connections = RNG.sample(others, min(n_connections, len(others)))

    # ── Purchase Orders ───────────────────────────────────────────
    purchase_orders: list[PurchaseOrder] = []
    for i in range(500):
        sup = RNG.choice(suppliers)
        days_ago = RNG.randint(0, 365)
        po_date = (now - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        base_amount = RNG.uniform(3_500, 490_000)
        is_anomaly = RNG.random() > 0.90    # ~10% anomaly rate

        if is_anomaly:
            anomaly_type = RNG.choice(ANOMALY_TYPES)
            # Anomalies have inflated amounts or quantities
            multiplier = RNG.uniform(2.5, 5.0) if anomaly_type == "price_spike" else 1.0
            amount = round(base_amount * multiplier, 2)
        else:
            anomaly_type = None
            amount = round(base_amount, 2)

        qty = RNG.randint(5, 1200)
        unit_price = round(amount / max(qty, 1), 2)

        po = PurchaseOrder(
            id=f"PO-2024-{i+1:05d}",
            supplier_id=sup.id,
            supplier_name=sup.name,
            category=RNG.choice(CATEGORIES),
            plant=RNG.choice(PLANTS),
            amount=amount,
            currency="EUR",
            date=po_date,
            status=RNG.choices(STATUSES, weights=PO_STATUS_WEIGHTS, k=1)[0],
            payment_days=RNG.randint(1, 92),
            quantity=qty,
            unit_price=unit_price,
            lead_time_actual=RNG.randint(2, 80),
            lead_time_planned=sup.lead_time_days,
            is_anomaly=is_anomaly,
            anomaly_type=anomaly_type,
            anomaly_score=0.0,   # filled by Isolation Forest
            invoice_match=RNG.random() > 0.07,
            three_way_match=RNG.random() > 0.13,
            approver=RNG.choice(APPROVERS),
            budget_code=f"BC-{RNG.randint(100,999)}",
        )
        purchase_orders.append(po)

    # ── Inventory ─────────────────────────────────────────────────
    comp_names = ["Alpha","Beta","Gamma","Delta","Omega","Sigma","Theta","Lambda"]
    inventory: list[InventorySKU] = []
    for i in range(120):
        sku = InventorySKU(
            sku=f"SKU-{i+1:06d}",
            description=f"Component {comp_names[i % len(comp_names)]}-{i//len(comp_names)+1}",
            category=CATEGORIES[i % len(CATEGORIES)],
            plant=PLANTS[i % len(PLANTS)],
            qty_on_hand=RNG.randint(0, 9500),
            reorder_pt=RNG.randint(80, 900),
            qty_on_order=RNG.randint(0, 4500),
            unit_cost=round(RNG.uniform(3.5, 220.0), 2),
            lead_time=RNG.randint(2, 45),
            abc_class=RNG.choices(["A","B","C"], weights=[0.20,0.30,0.50], k=1)[0],
            last_move=(now - timedelta(days=RNG.randint(0, 90))).strftime("%Y-%m-%d"),
            supplier_id=RNG.choice(sup_ids),
        )
        inventory.append(sku)

    # ── Monthly KPIs (24 months) ──────────────────────────────────
    monthly_kpis: list[MonthlyKPI] = []
    base_spend = 4_800_000.0
    for i in range(24):
        month_dt = now - timedelta(days=(23 - i) * 30)
        month_str = month_dt.strftime("%Y-%m")
        seasonal = math.sin(i * 0.52) * 900_000
        noise = RNG.uniform(-300_000, 300_000)
        spend = round(base_spend + seasonal + noise, 2)
        kpi = MonthlyKPI(
            month=month_str,
            spend=spend,
            pos=RNG.randint(16, 40),
            active_suppliers=RNG.randint(28, 52),
            savings=round(RNG.uniform(35_000, 310_000), 2),
            on_time_pct=round(RNG.uniform(0.76, 0.97), 3),
            anomaly_count=RNG.randint(2, 12),
            disputed_value=round(RNG.uniform(10_000, 400_000), 2),
        )
        monthly_kpis.append(kpi)

    return ERPDataset(
        suppliers=suppliers,
        purchase_orders=purchase_orders,
        inventory=inventory,
        monthly_kpis=monthly_kpis,
    )


# ── Singleton dataset (generated once at import) ──────────────────
_DATASET: Optional[ERPDataset] = None

def get_dataset() -> ERPDataset:
    global _DATASET
    if _DATASET is None:
        _DATASET = generate_erp_data()
    return _DATASET